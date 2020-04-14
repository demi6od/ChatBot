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

from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

BATCH_SIZE = 4
N_EPOCHS = 5
CLIP = 1
SEED = 1234

TRAIN_DIALOG = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# bert model
g_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
g_bert_vocab_size = g_bert_tokenizer.vocab_size
g_max_input_length = g_bert_tokenizer.max_model_input_sizes['bert-base-uncased']  # 512

g_bert = BertModel.from_pretrained('bert-base-uncased')
g_bert_emb_dim = g_bert.config.to_dict()['hidden_size']  # 768

# gpt model
g_gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
# g_gpt_tokenizer.add_special_tokens({'pad_token': g_bert_tokenizer.pad_token})
g_gpt_vocab_size = g_gpt_tokenizer.vocab_size

g_gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
g_gpt_emb_dim = g_gpt.config.to_dict()['n_embd']  # 768


SRC = Field(use_vocab=False,
            tokenize=g_bert_tokenizer.tokenize,
            preprocessing=g_bert_tokenizer.convert_tokens_to_ids,
            init_token=g_bert_tokenizer.cls_token_id,
            eos_token=g_bert_tokenizer.sep_token_id,
            pad_token=g_bert_tokenizer.pad_token_id,
            unk_token=g_bert_tokenizer.unk_token_id)

TGT = Field(use_vocab=False,
            tokenize=g_gpt_tokenizer.tokenize,
            preprocessing=g_gpt_tokenizer.convert_tokens_to_ids,
            init_token=g_gpt_tokenizer.bos_token_id,
            eos_token=g_gpt_tokenizer.eos_token_id,
            pad_token=g_gpt_tokenizer.eos_token_id,
            unk_token=g_gpt_tokenizer.unk_token_id)

g_data_fields = [('src', SRC), ('tgt', TGT)]

train_data, validation_data, test_data = TabularDataset.splits(
    path='datasets/',
    format='csv',
    train='chat_corpus_train.csv',
    validation='chat_corpus_validation.csv',
    test='chat_corpus_test.csv',
    skip_header=False,
    fields=g_data_fields)

g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.src),  # function used to group the data
    sort_within_batch=False,
    device=g_device)


class TransBertEncoder(nn.Module):
    def __init__(self, nhead=8, nlayers=6):
        super().__init__()

        # bert encoder
        self.bert = g_bert

        # transformer encoder, as bert last layer fine-tune
        encoder_layers = nn.TransformerEncoderLayer(d_model=g_bert_emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # src = [src len, batch size]

        with torch.no_grad():
            embedded = self.bert(src.transpose(0, 1))[0].transpose(0, 1)
            # embedded = [src len, batch size, emb dim]

        outputs = self.transformer_encoder(embedded)
        # outputs = [src len, batch size, hid dim * n directions]

        return outputs


class TransGptDecoder(nn.Module):
    def __init__(self, nhead=8, nlayers=6):
        super().__init__()

        # gpt decoder
        self.gpt = g_gpt

        # transformer decoder, as gpt first layer fine-tune
        decoder_layer = nn.TransformerDecoderLayer(d_model=g_gpt_emb_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

    def forward(self, meaning, tgt, output_len, teacher_forcing_ratio):
        # size of this batch
        batch_size = meaning.size(1)

        context = self.transformer_decoder(meaning, memory=meaning)
        # context = [meaning len, batch size, embedding dim]
        # tgt = [tgt len, batch size]

        # decide if we are going to use teacher forcing or not
        teacher_force = random.random() < teacher_forcing_ratio

        past = None
        predictions = torch.zeros(output_len, batch_size, g_gpt_vocab_size).to(g_device)
        for t in range(output_len):
            if t == 0:
                output, past = self.gpt(input_ids=None, inputs_embeds=context.transpose(0, 1), past=past)
                # output = [batch size, meaning len, vocab dim]
            else:
                if teacher_force and self.training:
                    context = tgt[t].unsqueeze(0)
                output, past = self.gpt(context.transpose(0, 1), past=past)
                # output = [batch size, 1, vocab dim]
            output = output.transpose(0, 1)
            predictions[t] = output[-1]
            token = torch.argmax(output[-1], 1)
            # token = [batch size]

            context = token.unsqueeze(0)
            # context = [1, batch size]

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
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, src, output_len, hidden):
        context = hidden
        # context = [1, batch size, hidden dim]

        # first input to the decoder is the <CLS> tokens
        fc_output = src[0].unsqueeze(0)

        # size of this batch
        batch_size = src.size(1)

        # tensor to store decoder outputs
        outputs = torch.zeros(output_len, batch_size, g_gpt_emb_dim).to(g_device)

        for t in range(0, output_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            gru_output, hidden = self.gru(fc_output, hidden)
            # gru_output = [1, batch size, hidden dim]

            emb_con = torch.cat((gru_output, context), dim=2)
            fc_output = self.fc(emb_con)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = fc_output
        return outputs


class DialogDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # ResNet, dropout on first 3 layers

        # input = self.dropout(input)
        #
        # output = input + F.relu(self.fc1(input))
        # output = self.dropout(output)
        #
        # output = output + F.relu(self.fc2(output))
        # output = self.dropout(output)
        #
        # output = output + self.fc3(output)  # no relu to keep negative values

        output = input + self.dropout(F.relu(self.fc1(input)))
        output = output + self.dropout(F.relu(self.fc2(output)))
        output = output + self.dropout(self.fc3(output))  # no relu to keep negative values

        return output


class Seq2Seq(nn.Module):
    def __init__(self, transbert_encoder, transgpt_decoder, gru_encoder, gru_decoder, dialog_dnn):
        super().__init__()

        self.transbert_encoder = transbert_encoder
        self.transgpt_decoder = transgpt_decoder

        self.gru_encoder = gru_encoder
        self.gru_decoder = gru_decoder

        self.dialog_dnn = dialog_dnn

    def forward(self, src, tgt, response_embeds_len, response_len, teacher_forcing_ratio):
        request_embeddings = self.transbert_encoder(src)
        request_meaning = self.gru_encoder(request_embeddings)

        if TRAIN_DIALOG:
            response_meaning = self.dialog_dnn(request_meaning)
        else:
            response_meaning = request_meaning

        response_embeddings = self.gru_decoder(request_embeddings, response_embeds_len, response_meaning)
        response = self.transgpt_decoder(response_embeddings, tgt, response_len, teacher_forcing_ratio)

        return response


INPUT_DIM = g_bert_vocab_size
OUTPUT_DIM = g_gpt_vocab_size
ENC_EMB_DIM = g_bert_emb_dim
DEC_EMB_DIM = g_gpt_emb_dim
HID_DIM = 2048  # 5 * 200
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TRANSFORMER_ENCODER_LAYER = 1  # bert fine-tune
TRANSFORMER_DECODER_LAYER = 1  # gpt fine-tune
TRANSFORMER_HEAD = 8

transbert_encoder = TransBertEncoder(TRANSFORMER_HEAD, TRANSFORMER_ENCODER_LAYER)
transgpt_decoder = TransGptDecoder(TRANSFORMER_HEAD, TRANSFORMER_DECODER_LAYER)
gru_encoder = GruEncoder(HID_DIM, ENC_EMB_DIM)
gru_decoder = GruDecoder(HID_DIM, DEC_EMB_DIM)
dialog_dnn = DialogDNN(HID_DIM, HID_DIM, HID_DIM)
g_model = Seq2Seq(transbert_encoder, transgpt_decoder, gru_encoder, gru_decoder, dialog_dnn).to(g_device)

