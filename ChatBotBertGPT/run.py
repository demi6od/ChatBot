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
    g_model.load_state_dict(torch.load('chatbot_transbertgpt-model.pt'))


# print chatbot's words
def print_chat(sentences):
    # sentences = [sentence len, batch size, vocab dim]
    print("chatbot: ", end="")

    sentence = g_gpt_tokenizer.decode(torch.argmax(sentences[:, 0, :], 1).tolist())
    print(sentence)


def print_src(sentences):
    # sentences = [sentence len, batch size]
    print("src: ", end="")

    sentence = g_bert_tokenizer.decode(sentences[:, 0].tolist())
    print(sentence)


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
        chat(g_model, test_iterator)


def chat(model, iterator):
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt
            teacher_forcing_ratio = 0  # turn off teacher forcing
            print(model)
            output = model(src,
                           tgt,
                           response_embeds_len=tgt.size(0)-1,
                           response_len=2*tgt.size(0)-1,
                           teacher_forcing_ratio=teacher_forcing_ratio)
            # tgt = [tgt len, batch size]
            # output = [tgt len, batch size, output dim]
            print(output)

            print_chat(output)
            print_src(src)
    return


if CHAT:
    talk(5)
