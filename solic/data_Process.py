import torch

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, BertModel, BertTokenizer,BertPreTrainedModel
import warnings
warnings.filterwarnings('ignore')

# label只有1和0，二分类任务，拼接level_1, level_2, level_3
# 不需要看两个句子的关联性
def load_data(data_dir):
    data = pd.read_csv(data_dir)
    data['content'].fillna('')
    data['text'] = data['content']+data['level_1']+data['level_2']+data['level_3']+data['level_4']
    return data

# 补全句长
def fill_paddings(data,maxlen):
    if len(data)<maxlen:
        pad_len = maxlen - len(data)
        pad_data = [0 for i in range(pad_len)]
        data = torch.tensor(data+pad_data)
    else:
        data = torch.tensor(data[:maxlen])
    return data

#bert的输入类型
#三部分组成，1、position embedding 2、segment embedding 3、word embedding
class InputDataset(Dataset):
    def __init__(self, data, tokenizer : BertTokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

        #获取bert的输入部分, index用来取数据的索引
    def __getitem__(self, index):
        data = str(self.data['text'][index])
        labels = self.data['label'][index]
        labels = torch.tensor(labels, dtype=torch.long)

        # 手动构建
        token = self.tokenizer.tokenize(data)
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        tokens_id = [101] + token_id + [102]
        input_id = fill_paddings(tokens_id, self.max_len)

        # mask部分
        attention_mask = [1 for _ in range(len(tokens_id))]
        mask = fill_paddings(attention_mask, self.max_len)

        #
        token_type_ids = [0 for _ in range(len(tokens_id))]
        token_type_ids = fill_paddings(token_type_ids, self.max_len)

        return {
            'text': data,
            'input_ids':input_id,
            'attention_mask': mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


if __name__ == "__main__":
    train_data = load_data('dataset/train.csv')
    test_data = load_data('dataset/test.csv')
    model_dir = 'dataset/bert-base-chinese/'
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    train_dataset = InputDataset(train_data,tokenizer,max_len=128)
    train_dataloader = DataLoader(train_dataset, batch_size=4)

    batch = next(iter(train_dataloader))

    pass