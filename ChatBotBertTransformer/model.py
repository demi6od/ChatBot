"""
Author: Chen Zhang (demi6od) <demi6d@gmail.com>
Date: 2020 Apr 14th
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.data import Field, BucketIterator, TabularDataset

import numpy as np

import random
import math

from transformers import BertTokenizer, BertModel

BATCH_SIZE = 64
N_EPOCHS = 4
CLIP = 1
SEED = 1234
TRAIN_DIALOG = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

g_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
g_vocab_size = g_tokenizer.vocab_size
g_max_input_length = g_tokenizer.max_model_input_sizes['bert-base-uncased']  # 512
g_bert = BertModel.from_pretrained('bert-base-uncased')
g_bert_emb_dim = g_bert.config.to_dict()['hidden_size']


def tokenize_and_cut(sentence):
    tokens = g_tokenizer.tokenize(sentence)
    tokens = tokens[:g_max_input_length-2]
    return tokens


SRC = Field(use_vocab=False,
            tokenize=tokenize_and_cut,
            preprocessing=g_tokenizer.convert_tokens_to_ids,
            init_token=g_tokenizer.cls_token_id,
            eos_token=g_tokenizer.sep_token_id,
            pad_token=g_tokenizer.pad_token_id,
            unk_token=g_tokenizer.unk_token_id)

TGT = Field(use_vocab=False,
            tokenize=tokenize_and_cut,
            preprocessing=g_tokenizer.convert_tokens_to_ids,
            init_token=g_tokenizer.cls_token_id,
            eos_token=g_tokenizer.sep_token_id,
            pad_token=g_tokenizer.pad_token_id,
            unk_token=g_tokenizer.unk_token_id)

g_data_fields = [('src', SRC), ('tgt', TGT)]

train_data, validation_data, test_data = TabularDataset.splits(
    path='datasets/',
    format='csv',
    train='chat_corpus_train.csv',
    validation='chat_corpus_validation.csv',
    test='chat_corpus_test.csv',
    skip_header=False,
    fields=g_data_fields)

# SRC.build_vocab(train_data, min_freq = 2, vectors="glove.6B.200d")
# TGT.build_vocab(train_data, min_freq = 2, vectors="glove.6B.200d")

g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.src),  # function used to group the data
    sort_within_batch=False,
    device=g_device)


# print chatbot's words
def print_chat(sentences):
    print("chatbot: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        # find one shot index from word embedding
        max_idx_t = word_embed.argmax()
        max_idx = max_idx_t.item()
        word = g_tokenizer.convert_ids_to_tokens(max_idx)
        print(word, end=" ")
    print("")  # new line at the end of sentence


def print_tgt(sentences):
    print("tgt: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        max_idx = word_embed.item()
        word = g_tokenizer.convert_ids_to_tokens(max_idx)
        print(word, end=" ")
    print("")  # new line at the end of sentence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransBertEncoder(nn.Module):
    def __init__(self, nhead=8, nlayers=6, dropout=0.5):
        super().__init__()

        # bert encoder
        self.bert = g_bert

        # transformer encoder, as bert last layer fine-tune
        self.pos_encoder = PositionalEncoding(g_bert_emb_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=g_bert_emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # src = [src len, batch size]

        with torch.no_grad():
            # embedded = [src len, batch size, emb dim]
            embedded = self.bert(src.transpose(0, 1))[0].transpose(0, 1)

        # embedded = self.pos_encoder(embedded)

        # src_mask = nn.Transformer().generate_square_subsequent_mask(len(embedded)).to(g_device)

        # outputs = [src len, batch size, hid dim * n directions]
        outputs = self.transformer_encoder(embedded)

        return outputs


class TransBertDecoder(nn.Module):
    def __init__(self, nhead=8, nlayers=6, dropout=0.5):
        super().__init__()

        # bert encoder
        self.bert = g_bert

        self.pos_decoder = PositionalEncoding(g_bert_emb_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=g_bert_emb_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

        self.fc_out = nn.Linear(g_bert_emb_dim, g_vocab_size)

    def forward(self, tgt, meaning, teacher_forcing_ratio):
        # tgt = [output_len, batch size]

        output_len = tgt.size(0)
        batch_size = tgt.size(1)
        # decide if we are going to use teacher forcing or not
        teacher_force = random.random() < teacher_forcing_ratio

        if teacher_force and self.training:
            tgt_emb_total = torch.zeros(output_len, batch_size, g_bert_emb_dim).to(g_device)

            for t in range(0, output_len):
                with torch.no_grad():
                    tgt_emb = self.bert(tgt[:t+1].transpose(0, 1))[0].transpose(0, 1)
                tgt_emb_total[t] = tgt_emb[-1]

            tgt_mask = nn.Transformer().generate_square_subsequent_mask(len(tgt_emb_total)).to(g_device)
            decoder_output = self.transformer_decoder(tgt=tgt_emb_total,
                                                      memory=meaning,
                                                      tgt_mask=tgt_mask)
            predictions = self.fc_out(decoder_output)
        else:
            # initialized the input of the decoder with sos_idx (start of sentence token idx)
            output = torch.full((output_len+1, batch_size), g_tokenizer.cls_token_id, dtype=torch.long, device=g_device)
            predictions = torch.zeros(output_len, batch_size, g_vocab_size).to(g_device)

            for t in range(0, output_len):
                with torch.no_grad():
                    tgt_emb = self.bert(output[:t+1].transpose(0, 1))[0].transpose(0, 1)

                # tgt_emb = [t, batch size, emb dim]
                # tgt_emb = self.pos_encoder(tgt_emb)

                tgt_mask = nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(g_device)

                # decoder_output = [t, batch size, emb dim]
                decoder_output = self.transformer_decoder(tgt=tgt_emb,
                                                          memory=meaning,
                                                          tgt_mask=tgt_mask)

                # prediction = [batch size, vocab size]
                prediction = self.fc_out(decoder_output[-1])

                # predictions = [output_len, batch size, vocab size]
                predictions[t] = prediction

                one_hot_idx = prediction.argmax(1)

                # output  = [output len, batch size]
                output[t+1] = one_hot_idx

        return predictions


class GruEncoder(nn.Module):
    """compress the request embeddings to meaning"""

    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        return hidden


class GruDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt, hidden):
        # first input to the decoder is the <CLS> tokens
        fc_output = src[0].unsqueeze(0)
        tgt_len = tgt.size(0)
        batch_size = tgt.size(1)

        # tensor to store decoder outputs
        outputs = torch.zeros(tgt_len, batch_size, g_bert_emb_dim).to(g_device)

        for t in range(0, tgt_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            gru_output, hidden = self.gru(fc_output, hidden)

            fc_output = self.fc(gru_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = fc_output
        return outputs


class DialogDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # ResNet, dropout on first 3 layers
        input = self.dropout(input)

        output = input + F.relu(self.fc1(input))
        output = self.dropout(output)

        output = output + F.relu(self.fc2(output))
        output = self.dropout(output)

        output = output + self.fc3(output)  # no relu to keep negative values

        return output


class Seq2Seq(nn.Module):
    def __init__(self, transbert_encoder, transbert_decoder, gru_encoder, gru_decoder, dialog_dnn):
        super().__init__()

        self.transbert_encoder = transbert_encoder
        self.transbert_decoder = transbert_decoder

        self.gru_encoder = gru_encoder
        self.gru_decoder = gru_decoder

        self.dialog_dnn = dialog_dnn

    def forward(self, src, tgt, teacher_forcing_ratio):
        request_embeddings = self.transbert_encoder(src)
        request_meaning = self.gru_encoder(request_embeddings)

        if TRAIN_DIALOG:
            response_meaning = self.dialog_dnn(request_meaning)
        else:
            response_meaning = request_meaning

        response_embeddings = self.gru_decoder(request_embeddings, tgt, response_meaning)
        response = self.transbert_decoder(tgt, response_embeddings, teacher_forcing_ratio)

        return response


INPUT_DIM = g_vocab_size
OUTPUT_DIM = g_vocab_size
ENC_EMB_DIM = g_bert_emb_dim
DEC_EMB_DIM = g_bert_emb_dim
HID_DIM = 2048  # 5 * 200
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TRANSFORMER_ENCODER_LAYER = 1  # bert fine-tune
TRANSFORMER_DECODER_LAYER = 3  # semantics -> morphology -> syntax
TRANSFORMER_HEAD = 8

transbert_encoder = TransBertEncoder(TRANSFORMER_HEAD, TRANSFORMER_ENCODER_LAYER, ENC_DROPOUT)
transbert_decoder = TransBertDecoder(TRANSFORMER_HEAD, TRANSFORMER_DECODER_LAYER, DEC_DROPOUT)
gru_encoder = GruEncoder(HID_DIM, ENC_EMB_DIM)
gru_decoder = GruDecoder(HID_DIM, DEC_EMB_DIM)
dialog_dnn = DialogDNN(HID_DIM, HID_DIM, HID_DIM)
g_model = Seq2Seq(transbert_encoder, transbert_decoder, gru_encoder, gru_decoder, dialog_dnn).to(g_device)
