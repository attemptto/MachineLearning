import os
from transformers import BertModel,BertTokenizer,AdamW
from torch.utils.data import DataLoader,Dataset
import torch
from seqeval.metrics import accuracy_score,f1_score,precision_score,recall_score

def load_data(file_path):
    with open(os.path.join('dataset',file_path+'.txt'),encoding='utf-8') as f:
        all_data = f.read().split('\n')
        all_text = []
        all_label = []

        text = []
        label = []
        for data in all_data:
            if data == '':
                all_text.append(text)
                all_label.append(label)
                text=[]
                label=[]
            else:
                t,l = data.split(' ')
                text.append(t)
                label.append(l)

    return all_text,all_label

def build_label(text_label):
    label_2_index = {'PAD':0,'UNK':1}
    for sentence in text_label:
        for word in sentence:
            if word not in label_2_index:
                label_2_index[word] = len(label_2_index)
    return label_2_index,list(label_2_index)



class BertDataset(Dataset):
    def __init__(self, all_text, all_label,label_2_index, tokenizer, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.label_2_index = label_2_index
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]

        text_index = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_len+2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
           )
        label_index = [0] + [self.label_2_index.get(l, 1) for l in label] + [0] + [0] * (self.max_len - len(text))
        label_index = torch.tensor(label_index)
        return text_index.reshape(-1), label_index, len(label)

    def __len__(self):
        return len(self.all_text)

class MyModel(torch.nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(768,class_num)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        bert_out = self.bert(batch_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # 0:字符级别特征， 1:句子级别特征
        pre = self.classifier(bert_out0)

        if batch_label is not None:
            loss = self.loss_func(pre.reshape(-1,pre.shape[-1]), batch_label.reshape(-1))
            return loss

        else:
            return torch.argmax(pre,dim=-1)




if __name__=="__main__":
    train_text, train_label = load_data('train')
    dev_text, dev_label = load_data('dev')
    test_text, test_label = load_data('test')

    batch_size = 20
    epoch = 100
    max_len = 30
    lr = 0.0001

    label_2_index,index_2_label = build_label(train_label)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_Dataset = BertDataset(train_text,train_label,label_2_index,tokenizer,max_len)
    train_DataLoader = DataLoader(train_Dataset,batch_size=batch_size, shuffle=False)

    dev_Dataset = BertDataset(dev_text, dev_label, label_2_index, tokenizer, max_len)
    dev_DataLoader = DataLoader(dev_Dataset, batch_size=batch_size, shuffle=False)


    model = MyModel(len(label_2_index)).to(device)
    opt = AdamW(params=model.parameters(),lr=lr)
    for e in range(epoch):
        model.train()
        for train_batch_text,train_batch_label,train_batch_len in train_DataLoader:
            train_batch_text = train_batch_text.to(device)
            train_batch_label = train_batch_label.to(device)
            loss = model.forward(batch_index=train_batch_text,batch_label=train_batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

            print(f'loss:{loss:.2f}')
            break

        model.eval()
        all_pre = []
        all_tag = []
        for dev_batch_text,dev_batch_label,dev_batch_len in dev_DataLoader:
            dev_batch_text = dev_batch_text.to(device)
            dev_batch_label = dev_batch_label.to(device)
            pre = model.forward(batch_index=dev_batch_text)

            pre = pre.cpu().numpy().tolist()
            #pre = [[index_2_label[j] for j in i] for i in pre]
            dev_batch_label = dev_batch_label.cpu().numpy().tolist()
            #dev_batch_label = [[index_2_label[j] for j in i] for i in dev_batch_label]
            for p,t,l in zip(pre,dev_batch_label,dev_batch_len):
                p = p[1:1+l]
                t = t[1:1+l]

                p = [index_2_label[i] for i in p]
                t = [index_2_label[i] for i in t]

                all_pre.append(p)
                all_tag.append(t)

        f1_score = f1_score(all_tag,all_pre)
    pass