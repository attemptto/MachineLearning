import os
import pandas as pd
import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_data(train_or_test, nums=None):
    with open(os.path.join('dataset', train_or_test + '.txt'), encoding="utf-8") as f:
        all_data = f.read().split('\n')

        texts = []
        labels = []
        for text in all_data:
            if text:
                t, l = text.split('\t')
                texts.append(t)
                labels.append(l)
        if nums is None:
            return texts, labels
        else:
            return texts[:nums], labels[:nums]


def build_corpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    # onehot return word_2_index, np.eye(len(word_2_index), dtype=np.float32)
    return word_2_index, np.random.normal(0,1,size=(len(word_2_index),embedding_num)).astype(np.float32)

def load_stop_words(file='stop_words.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


class OhDataset(Dataset):
    def __init__(self, texts, labels, word_2_index, index_2_onehot, max_len):
        self.texts = texts
        self.labels = labels
        self.word_2_index = word_2_index
        self.index_2_onehot = index_2_onehot
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.texts[index]
        label = int(self.labels[index])

        # 裁剪数据长度
        text = text[:self.max_len]
        # 取出这句话中每个词的index
        text_index = [word_2_index.get(i, 1) for i in text]
        # 每个句子都定长，不够则补全
        text_index = text_index + [0] * (self.max_len - len(text_index))
        # 取出每个词的 onehot 向量
        text_onehot = self.index_2_onehot[text_index]

        return text_onehot, label

    def __len__(self):
        return len(self.labels)


class OhModel(nn.Module):
    def __init__(self, embedding_num, hidden_num, class_num, max_len):
        super().__init__()
        self.rnn = nn.RNN(embedding_num, hidden_num, num_layers=1, bias=False,batch_first=True)
        self.linear1 = nn.Linear(hidden_num, class_num)

    def forward(self, text_onehot, labels=None):
        output, h_n = self.rnn(text_onehot,None)


def test_file():
    global model, word_2_index, index_2_onehot, device, max_len

    test_texts, test_labels = load_data("test")

    test_dataset = OhDataset(test_texts, test_labels, word_2_index, index_2_onehot, max_len)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    result = []

    for text, label in test_loader:
        text = text.to(device)
        model(text)
        result.extend(model.pre)
    with open(os.path.join("dataset", "test_res.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(i) for i in result]))
    test_acc = sum([i == int(j) for i, j in zip(result, test_labels)]) / len(test_labels)
    print(f"test acc:{test_acc * 100 :.2f}%")
    # print(f"dev acc:{right_num / len(dev_labels) * 100 :.2f} %")


# def read_data(train_or_test):
#     with open(os.path.join("dataset",train_or_test + '.csv'), encoding='utf-8') as f:
#         all_data = f.read().split('\n')
#     all_data = all_data[1:]
#     texts = []
#     labels = []
#     for data in all_data:
#         if data:
#             _, text, label = data.split("\t")
#             texts.append(text)
#             labels.append(label)
#
#     return texts, labels


if __name__ == "__main__":

    train_texts, train_labels = load_data('train')
    dev_texts, dev_labels = load_data('dev')

    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lr = 0.0006
    epoch = 5
    # stop_words = load_stop_words()
    batch_size = 60
    max_len = 25
    hidden_num = 30
    embedding_num = 200
    class_num = len(set(train_labels))

    word_2_index, index_2_onehot = build_corpus(train_texts,embedding_num)
    # dev_word_2_index, dev_index_2_onehot = build_corpus(dev_texts)
    # dev_class_num = len(set(goods_labels))

    train_dataset = OhDataset(train_texts, train_labels, word_2_index, index_2_onehot, max_len)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=False)

    dev_dataset = OhDataset(dev_texts, dev_labels, word_2_index, index_2_onehot, max_len)
    dev_dataLoader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = OhModel(embedding_num, hidden_num, class_num, max_len)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr)

    for _ in range(epoch):
        for texts, labels in tqdm(train_dataLoader):
            texts = texts.to(device)
            labels = labels.to(device)

            loss = model(texts, labels)
            loss.backward()

            optim.step()
            optim.zero_grad()

        right_num = 0
        for texts, labels in dev_dataLoader:
            texts = texts.to(device)
            model(texts)
            right_num += int(sum([i == j for i, j in zip(model.pre, labels)]))
        # print(f"dev acc:{right_num/len(dev_labels) * 100 :.2f} %")
test_file()
