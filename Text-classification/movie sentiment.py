# %%

import torch
import torchvision
import numpy as np
import tokenizers
import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import re
from torch.utils.data import Dataset, DataLoader


# %%

# len = 15
from tqdm import tqdm


def getNgrams(input, n):
    output = {}
    for i in range(len(input) - n + 1):
        ngramTemp = " ".join(input[i:i + n])
        if ngramTemp not in output:
            output[ngramTemp] = 0
        output[ngramTemp] += 1
    return output


def build_corpus(input, embedding_num, max_len):
    dict_set = set()
    for phrase in input:
        words = phrase.split(' ')
        for word in words:
            if word != ' ' not in dict_set:
                dict_set.add(word)
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    dic_text = dict(dict_list)
    embedding_matrix = []
    for phrase in input:
        sentence = []
        words = phrase.split(' ')
        for word in words:
            if word != ' ':
                current_index = dic_text[word]
                sentence.append(current_index)
        if len(sentence) < max_len:
            sentence = sentence + [0] * (max_len - len(sentence))
        else:
            sentence = sentence[:max_len]
        embedding_matrix.append(sentence)
    return dict_list, embedding_matrix


def clean_data(input):
    after_clean = []
    for phrase in input:
        after = re.sub("[^a-zA-Z]", " ", phrase)
        after_clean.append(after)
    return after_clean


train_text = pd.read_csv('dataset2/train.tsv', sep='\t')['Phrase']
train_data = clean_data(train_text)
word_2_index, embedding_matrix = build_corpus(train_data, 100, 36)

embedding_matrix = pd.DataFrame(embedding_matrix)

sentiment = pd.read_csv('dataset2/train.tsv', sep='\t')['Sentiment']
# sentiment = np.array(sentiment)
batch_size = 10

data = pd.concat((embedding_matrix, sentiment), axis=1)

train_text = embedding_matrix
train_label = data['Sentiment']
train_text = np.array(train_text)
train_label = np.array(train_label)

class Mydataset(Dataset):
    def __init__(self, text, label, max_len):
        self.text = text
        self.label = label
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.text[index]
        label = int(self.label[index])
        return text, label

    def __len__(self):
        return len(self.text)


class Mymodel(torch.nn.Module):
    def __init__(self, embedding_num, hidden_num, class_num):
        super().__init__()
        self.linear1 = torch.nn.Linear(embedding_num, hidden_num)
        self.linear2 = torch.nn.Linear(hidden_num, class_num)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, text_onehot, labels):
        hidden1 = self.linear1(text_onehot)
        hidden2 = self.linear1(hidden1)
        loss = self.loss_func(hidden2,labels)
        return loss


dataset = Mydataset(train_text, train_label, max_len=36)
dataloader = DataLoader(dataset, batch_size=batch_size)

model = Mymodel(embedding_num=36,hidden_num=20,class_num=4)

for _ in range(10):
    for texts, labels in tqdm(dataloader):
        loss = model(texts, labels)
        loss.backward()
pass
