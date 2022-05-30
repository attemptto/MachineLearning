from data_Process import load_data, InputDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
import numpy as np

# 文本的分类
class BertForSeq(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels = None, return_dict = None):

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids , attention_mask,token_type_ids,  return_dict)
        pool_output = outputs[1]
        pool_output = self.dropout(pool_output)
        # logits -—— softmax层的输入（0.4， 0.6）--- 1
        logits = self.classifier(pool_output)

        loss = None
        if labels is not None:
            loss_fact  = nn.CrossEntropyLoss()
            loss = loss_fact(logits.view(-1,self.num_labels),labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

if __name__ == '__main__':

    ## 加载编码器和模型
    tokenizer = BertTokenizer.from_pretrained('dataset/bert-base-chinese')
    model = BertForSeq.from_pretrained('dataset/bert-base-chinese')
    ## 准备数据
    dev = load_data('dataset/dev.csv')
    dev_dataset = InputDataset(dev,tokenizer=tokenizer,max_len=128)
    dev_dataloader = DataLoader(dev_dataset,batch_size=4,shuffle=False)
    ## 把数据做成batch
    batch = next(iter(dev_dataloader))
    ## 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 输入embedding
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['labels'].to(device)
    ## 预测
    model = model.to(device)
    model.eval()
    ## 得到输出
    outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
    ## 取输出里面的loss和logits
    pre = torch.argmax(outputs[1], dim=1)
    pass