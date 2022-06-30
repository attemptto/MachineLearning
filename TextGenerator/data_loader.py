import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader,Dataset
import pickle
import os
import torch
from torch import nn


# 划分每一个字
def split_poetry(poetry):
    texts = pd.read_table(poetry,header=None)
    texts.columns = ['text']
    texts = texts.text
    data = []
    for text in texts:
        sentence = []
        for word in text:
            word = " ".join(word)
            sentence.append(word)
        data.append(sentence)
    df = pd.DataFrame(data)
    df.to_csv("dataset\split_poetry7.txt",sep='\t',index=None,header=None)
    return data

def train_vec(origin_text,split_text):
    # word2vec包
    if os.path.exists(split_text)==False:
        all_data = split_poetry(origin_text)
    all_data = pd.read_table('dataset/split_poetry7.txt',index_col=None,header=None)
    vec_params = 'vec_params.pkl'

    if os.path.exists(vec_params):
        return all_data,pickle.load(open(vec_params,'rb'))
    model = Word2Vec(all_data,vector_size=100,min_count=1,workers=5)
    pickle.dump([model.syn1neg, model.wv.key_to_index,model.wv.index_to_key],open(vec_params,'wb'))
    return all_data,(model.syn1neg, model.wv.key_to_index,model.wv.index_to_key)

class MyWord2VecModel(nn.Module):
    
    def __init__(self,embedding_num,hidden_num,corpus_num):
        super().__init__()
        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.corpus_num = corpus_num

        self.lstm = nn.LSTM(input_size=embedding_num,hidden_size=hidden_num,batch_first=True,num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0,1)
        self.linear1 = nn.Linear(hidden_num,corpus_num)
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self,xs_embedding,h_0=None,c_0=None):
        xs_embedding= xs_embedding.to(device)
        if h_0 == None and c_0 == None:
            h_0 = torch.tensor(np.zeros((2,xs_embedding.shape[0],self.hidden_num),dtype=np.float32))
            c_0 = torch.tensor(np.zeros((2,xs_embedding.shape[0],self.hidden_num),dtype=np.float32))
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        hidden,(h_0,c_0)=self.lstm(xs_embedding,(h_0,c_0))
        hidden_drop = self.dropout(hidden)
        flatten_hidden = self.flatten(hidden_drop)
        pre = self.linear1(flatten_hidden)
        return pre,(h_0,c_0)


class MyWord2Vec(Dataset):
    def __init__(self,all_data,w1,word_2_index):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        one_sen = self.all_data[index]
        index_sen = [self.word_2_index[word] for word in one_sen]
        xs_index = index_sen[:-1]
        ys_index = index_sen[1:]

        xs_embedding = self.w1[xs_index]

        return xs_embedding, np.array(ys_index).astype(np.int64)

    def __len__(self,):
        return len(self.all_data)


def generate_poetry():
    res = ""
    word_index = np.random.randint(0,word_size,1)[0]
    res += index_2_word[word_index]
    h_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))
    c_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))
    for i in range(31):
        word_embedding = torch.tensor(w1[word_index][None][None])
        pre,(h_0,c_0)= model(word_embedding,h_0,c_0)
        word_index = int(torch.argmax(pre))
        res += index_2_word[word_index]
    print(res)


if __name__ == "__main__":
    #train_txt = split_poetry('dataset/poetry7.txt')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_data,(w1,word_2_index,index_2_word) = train_vec('dataset/poetry7.txt','dataset/split_poetry7')
    all_data = np.array(all_data)
    dataset = MyWord2Vec(all_data,w1=w1,word_2_index=word_2_index)
    dataloader = DataLoader(dataset,batch_size=5,shuffle=True)
    word_size, embedding_num = w1.shape

    hidden_num = 51
    model = MyWord2VecModel(embedding_num,hidden_num,word_size)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

    epochs = 50

    for epoch in range(epochs):
        for batch_index,(xs_embedding, ys_index) in enumerate(dataloader):
            xs_embedding = xs_embedding.to(device)
            ys_index = ys_index.to(device)

            pre, _ = model(xs_embedding)
            loss = model.crossEntropy(pre,ys_index.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 5 ==0:
                print(f'loss:{loss:.3f}')
                generate_poetry()