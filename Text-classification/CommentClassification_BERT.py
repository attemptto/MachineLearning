import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer,BertModel,AdamW


def load_data(data_path):
    all_data = pd.read_csv(data_path)
    return all_data

def fill_sentence(data,max_len):
    res = []
    for sentence in data:
        cur_sentence = []
        if len(sentence)< max_len:
            for word in sentence:
                cur_sentence.append(word)
            pad_len = max_len - len(sentence)
            for i in range(pad_len):
                cur_sentence.append([0])
            res.append(cur_sentence)
        else:res.append(sentence[:max_len])
    return res

def bert_encoder(data):
    text = data['text']
    label = data['label']
    text_encoder = token.batch_encode_plus(
        batch_text_or_text_pairs=text,
        truncation=True,
        padding='max_length',
        max_length=500,
        return_tensors='pt',
        return_length=True
    )
    input_ids = text_encoder['input_ids']
    attention_mask = text_encoder['attention_mask']
    token_type_ids = text_encoder['token_type_ids']
    label = torch.LongTensor(label).to(device)

    return input_ids, attention_mask, token_type_ids, label

class Mydataset(Dataset):
    def __init__(self,input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        token_type_ids = self.token_type_ids[index]
        label = self.label[index]
        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.input_ids)

class Mymodel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768,2)

    def forward(self,input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pre_trained(input_ids,attention_mask,token_type_ids)
            out = self.linear(out.last_hidden_state[:, 0])
            out = out.softmax(dim=1)
            return out

if __name__=="__main__":
    global train_text,train_label

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    all_data = load_data("F:\Data\dataset\hotel_comment\ChnSentiCorp.csv")
    # 自己编码所有句子
    #fill_sen = pd.DataFrame(fill_sentence(train_text,128))
    token = BertTokenizer.from_pretrained('bert-base-chinese')
    input_ids, attention_mask, token_type_ids, label = bert_encoder(data=all_data)
    dataset = Mydataset(input_ids, attention_mask, token_type_ids, label)
    data_loader = DataLoader(dataset,batch_size=16,shuffle=False,drop_last=True)



    pre_trained = BertModel.from_pretrained('bert-base-chinese')
    pre_trained = pre_trained.to(device)
    # out = pre_trained(input_ids=input_ids,
    #   attention_mask=attention_mask,
    #   token_type_ids=token_type_ids)

    model = Mymodel()
    model = model.to(device)

    optimizer = AdamW(model.parameters(),lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(data_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        loss = criterion(out,label)
        loss.requires_grad_(True)
        loss.backward()
        if i % 5 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == label).sum().item() / len(label)

            print(i, loss.item(), accuracy)

        if i == 300:
            break
