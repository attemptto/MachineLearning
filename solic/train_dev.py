import time
import numpy as np
from torch import nn
import time
import os
import torch
import logging
from torch.optim import Adam
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from bert_Model import BertForSeq
from data_Process import InputDataset, load_data, fill_paddings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(batch_size, epochs):
    model = BertForSeq.from_pretrained('bert-base-chinese')

    train = load_data('dataset/train.csv')
    dev = load_data('dataset/dev.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 数据处理 word embedding+ segment embedding + position embedding
    train_dataset = InputDataset(train,tokenizer,max_len=128)
    dev_dataset = InputDataset(train, tokenizer, max_len=128)
    # dataloader
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    #优化器
    optimizer = Adam(model.parameters(),lr=2e-5)
    total_step = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0,num_training_steps=total_step)

    total_time = time.time()

    log = log_creater(output_dir='./cache/logs/')



def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)
