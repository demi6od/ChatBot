"""
Author: Chen Zhang (demi6od) <demi6d@gmail.com>
Date: 2020 Apr 14th
"""

from train import *
import csv

TRAIN = True
CHAT = True

if TRAIN:
    train_epoch(N_EPOCHS)

if CHAT:
    g_model.load_state_dict(torch.load('chatbot_transbert-model.pt'))


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
            fields=g_data_fields)
        test_iterator = BucketIterator(
            test_data,
            batch_size=BATCH_SIZE,
            sort_key=lambda x: len(x.src),
            sort_within_batch=False,
            device=g_device)
        chat(g_model, test_iterator, criterion)


def chat(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt

            teacher_forcing_ratio = 0  # turn off teacher forcing
            output = model(src, tgt[0:-1], teacher_forcing_ratio)

            print_chat(output)
            print_tgt(tgt)

            # tgt = [tgt len, batch size]
            # output = [tgt len, batch size, output dim]

            output_dim = output.shape[-1]

            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]
            loss = criterion(output.view(-1, output_dim), tgt[1:].view(-1))

            print(f'\t Val. Loss: {loss:.3f} |  Val. PPL: {math.exp(loss):7.3f}')

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if CHAT:
    talk(5)


