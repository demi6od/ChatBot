"""
Author: Chen Zhang (demi6od) <demi6d@gmail.com>
Date: 2020 Apr 14th
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator, TabularDataset

import spacy
import numpy as np

import random
import math
import time

import csv

BATCH_SIZE = 128
N_EPOCHS = 10
CLIP = 1
SEED = 1234
PRINT_CHAT = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

data_fields = [('src', SRC), ('trg', TRG)]

train_data, validation_data, test_data = TabularDataset.splits(
    path='datasets/',
    format='csv',
    train='chat_corpus_train.csv',
    validation='chat_corpus_validation.csv',
    test='chat_corpus_test.csv',
    skip_header=False,
    fields=data_fields)

SRC.build_vocab(train_data, min_freq = 2, vectors="glove.6B.200d")
TRG.build_vocab(train_data, min_freq = 2, vectors="glove.6B.200d")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.src),  # function used to group the data
    sort_within_batch=False,
    device=device)


# print chatbot's words
def print_chat(sentences):
    print("chatbot: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        # find one shot index from word embedding

        # one_shot_idx = 0
        # for idx in range(len(word_embed)):
        #     if word_embed[idx] > word_embed[one_shot_idx]:
        #         one_shot_idx = idx

        max_idx_t = word_embed.argmax()
        max_idx = max_idx_t.item()
        word = TRG.vocab.itos[max_idx]
        print(word, end=" ")
    print("")  # new line at the end of sentence


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        # use pre-trained word embeddings
        self.embedding = nn.Embedding(input_dim, emb_dim)
        weight_matrix = SRC.vocab.vectors
        self.embedding.weight.data.copy_(weight_matrix)

        self.rnn = nn.GRU(emb_dim, hid_dim)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.embedding(src)

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)  # no cell state!

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        # use pre-trained word embeddings
        self.embedding = nn.Embedding(output_dim, emb_dim)
        weight_matrix = TRG.vocab.vectors
        self.embedding.weight.data.copy_(weight_matrix)

        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.embedding(input)

        # embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)
        output = self.dropout(output)  # dropout the fc layer

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.8):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        if PRINT_CHAT:
            print_chat(outputs)
        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 200
DEC_EMB_DIM = 200
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)


def freeze_layers(layers):
    for layer in layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False


# freeze the embedding parameters
freeze_layers([model.encoder.embedding, model.decoder.embedding])


def init_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            nn.init.normal_(param.data, mean=0, std=0.01)


init_weights(model)

# print the model structure
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(n_epochs):
    best_validation_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        #validation_loss = evaluate(model, validation_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if validation_loss < best_validation_loss:
        #     best_validation_loss = validation_loss
        #     torch.save(model.state_dict(), 'chatbot_rnn-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        #print(f'\t Val. Loss: {validation_loss:.3f} |  Val. PPL: {math.exp(validation_loss):7.3f}')

    torch.save(model.state_dict(), 'chatbot_rnn-model.pt')


# train_epoch(N_EPOCHS)

model.load_state_dict(torch.load('chatbot_rnn-model.pt'))

# test_loss = evaluate(model, test_iterator, criterion)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

def create_chat_csv(filename, sentence):
    f_csv = open("datasets/" + filename + '.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow([sentence, sentence])


def talk(num):
    for i in range(num):
        user_input = input("user: ")
        create_chat_csv("chat", user_input)
        test_data = TabularDataset(
            path='datasets/chat.csv',
            format='csv',
            skip_header=False,
            fields=data_fields)
        test_iterator = BucketIterator(
            test_data,
            batch_size=BATCH_SIZE,
            sort_key=lambda x: len(x.src),
            sort_within_batch=False,
            device=device)
        evaluate(model, test_iterator, criterion)


talk(5)
