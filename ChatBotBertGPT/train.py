"""
Author: Chen Zhang (demi6od) <demi6d@gmail.com>
Date: 2020 Apr 14th
"""

from model import *

import torch.optim as optim
import math
import time

CONTINUE_TRAIN = True


def freeze_layers(layers):
    for layer in layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False


def freeze_params(model, keywords):
    for name, param in model.named_parameters():
        for keyword in keywords:
            if name.find(keyword) != -1:
                param.requires_grad = False


def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


# freeze the bert and gpt parameters
freeze_params(g_model, ['bert.', 'gpt.'])
# freeze_layers([model.encoder.embedding, model.decoder.embedding])


def init_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            nn.init.normal_(param.data, mean=0, std=0.01)
            # nn.init.uniform_(param.data, -0.1, 0.1)


init_weights(g_model)

# print the model structure
# print(g_model)
# print_params(g_model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(g_model):,} trainable parameters')

g_optimizer = optim.Adam(g_model.parameters())

g_criterion = nn.CrossEntropyLoss(ignore_index=g_gpt_tokenizer.eos_token_id)
# g_criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    g_bert.eval()
    g_gpt.eval()

    print("Batch num: " + str(len(iterator)))

    epoch_loss = 0
    for i, batch in enumerate(iterator):
        # print the batch index
        if i % 50 == 0:
            print("")
        print(i, end=",")

        src = batch.src
        tgt = batch.tgt
        optimizer.zero_grad()
        teacher_forcing_ratio = 1
        output = model(src,
                       tgt,
                       response_embeds_len=tgt.size(0)-1,
                       response_len=tgt.size(0)-1,
                       teacher_forcing_ratio=teacher_forcing_ratio)
        # tgt = [tgt len, batch size]
        # output = [tgt len, batch size, output dim]

        output_dim = output.shape[-1]
        loss = criterion(output.view(-1, output_dim), tgt[1:].view(-1))
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
            tgt = batch.tgt
            teacher_forcing_ratio = 0  # turn off teacher forcing
            output = model(src,
                           tgt,
                           response_embeds_len=tgt.size(0) - 1,
                           response_len=tgt.size(0) - 1,
                           teacher_forcing_ratio=teacher_forcing_ratio)
            # tgt = [tgt len, batch size]
            # output = [tgt len, batch size, output dim]

            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), tgt[1:].view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(n_epochs):
    # best_validation_loss = float('inf')

    if CONTINUE_TRAIN:
        g_model.load_state_dict(torch.load('chatbot_transbertgpt-model.pt'))
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(g_model, train_iterator, g_optimizer, g_criterion, CLIP)
        # validation_loss = evaluate(model, validation_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if validation_loss < best_validation_loss:
        #     best_validation_loss = validation_loss
        #     torch.save(model.state_dict(), 'chatbot_rnn-model.pt')

        print(f'\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        #print(f'\t Val. Loss: {validation_loss:.3f} |  Val. PPL: {math.exp(validation_loss):7.3f}')

        torch.save(g_model.state_dict(), 'chatbot_transbertgpt-model.pt')

