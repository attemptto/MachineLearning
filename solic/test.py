#%%
import transformers
import torch.nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import shutil

train_data= pd.read_csv('dataset/train.csv',sep='\t',header=None,index_col=None)
dev_data= pd.read_csv('dataset/dev.csv',sep='\t',header=None,index_col=None)

train_data.head()
#%%
cols = ['text','label']
train_data.columns = cols
dev_data.columns = cols
train_text  = train_data['text']
train_label = train_data['label']
dev_text = dev_data['text']
dev_label = dev_data['label']
#%%

MAX_LEN = 20
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
#%%
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#%%
example_text = "今天早上，你吃饭了吗？"
encodings = tokenizer.encode_plus(
    example_text,
    add_special_tokens=True,
    max_length= MAX_LEN,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

encodings
#%%

class CustomDataset(Dataset):
    def __init__(self,text,label,tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.text = text
        self.label = label
        self.max_len = maxlen

    def __getitem__(self, index):
        text = self.text[index]
        label = self.label[index]

        input = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length= self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids':input['input_ids'].flatten(),
            'attention_mask':input['attention_mask'].flatten(),
            'token_type_ids':input['token_type_ids'].flatten(),
            'label':torch.tensor(self.label[index])
        }
    def __len__(self):
        return len(self.text)

#%%
train_dataset = CustomDataset(train_text,train_label,tokenizer,maxlen=MAX_LEN)
dev_dataset = CustomDataset(dev_text,dev_label,tokenizer,maxlen=MAX_LEN)

#%%

train_data_loader = DataLoader(train_dataset,batch_size=3,shuffle=True)
dev_data_loader = DataLoader(dev_dataset,batch_size=3,shuffle=True)


#%%
device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
# 加载权重文件
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
#%%

class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass,self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 25)

    def forward(self,input_ids,attention_mask,token_type_ids):
        output = self.bert_model(input_ids,attention_mask,token_type_ids)
        output = self.dropout(output.pooler_output)
        output = self.linear(output)
        return output

model = BertClass()
model.to(device)
#%%
def loss_fn(outputs,targets):
    return torch.nn.CrossEntropyLoss(outputs, targets.float())

optimizer = torch.optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

#%%

val_targets=[]
val_outputs=[]

#%%
def train_model(n_epochs,train_loader,dev_loader,model,optimizer):
    valid_loss_min = np.Inf

    for epoch in range(1,n_epochs+1):
        train_loss= 0
        dev_loss = 0
        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(train_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            labels = data['label'].to(device)

            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            #outputs = outputs.reshape(-1)
            #loss = loss_fn(outputs,labels)
            cross_entropy = torch.nn.CrossEntropyLoss()
            loss = cross_entropy(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('before loss data in training', loss.item(), train_loss)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        print('############# Epoch {}: Training End     #############'.format(epoch))

#%%
train_model = train_model(EPOCHS,train_loader=train_data_loader,dev_loader=dev_data_loader,model=model,optimizer=optimizer)

