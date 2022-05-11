import pickle
import numpy as np
import os
import pandas as pd
import jieba
#w1,word_2_index, index_2_word, w2 = pickle.load(open('word2vec.pkl','rb'))

#%%

def word_voc(word):
    return w1[word_2_index[word]]

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis=1,keepdims=True)

#%%

def load_stop_words(file='stop_words.txt'):
    with open(file,'r',encoding='utf-8') as f:
        return f.read().split('\n')

#%%

def cut_words(file='1.csv'):
    stop_words = load_stop_words()
    all_data = pd.read_csv(file,encoding="gbk",names = ["data"])["data"]
    result = []
    for words in all_data:
        c_words = jieba.lcut(words)
        result.append([word for word in c_words if word not in stop_words])
    return result



#%%

def get_dic(data):
    index_2_word = []
    #加入到Index_2_word表中
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)
    #每个单词的索引
    word_2_index = {word:index for index,word in enumerate(index_2_word)}
    word_size = len(word_2_index)
    word_2_onehot = {}
    for word,index in word_2_index.items():
        one_hot = np.zeros((1,word_size))
        one_hot[0,index] = 1
        word_2_onehot[word] = one_hot

    return word_2_index,index_2_word,word_2_onehot


#%%

data = cut_words()
word_2_index,index_2_word,word_2_onehot = get_dic(data)
print(word_2_index,index_2_word,word_2_onehot)


#%%

#训练模型
embedding_num = 108
lr = 0.01
epochs = 10
n_gram = 3
word_size = len(word_2_index)
w1 = np.random.normal(-1,1,size=(word_size,embedding_num))
w2 = np.random.normal(-1,1,size=(embedding_num,word_size))

for e in range(epochs):
    for words in data:
        for index_now_word, now_word in enumerate(words):
            #获取当前词的one-hot
            now_word_vec = word_2_onehot[now_word]
            #截断
            other_words = words[max(index_now_word-n_gram, 0):index_now_word] + words[index_now_word+1:index_now_word+1+n_gram]
            for other_word in other_words:
                other_word_vec = word_2_onehot[other_word]

                hidden = now_word_vec @ w1
                p = hidden @ w2
                pre = softmax(p)

                #loss交叉熵损失函数

                G2 = pre - other_word_vec
                delta_w2 = hidden.T @ G2
                G1 =  G2@w2.T
                delta_w1 = now_word_vec.T@G1

                w1 -= lr*delta_w1
                w2 -= lr*delta_w2

with open("word2vec.pkl","wb") as f:
    pickle.dump([w1,word_2_index,index_2_word],f)  #负采样
