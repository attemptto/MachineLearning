import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, dk_dim, n_heads):
        super().__init__()
        self.atte_layers = nn.ModuleList([OneHeadAttention(input_dim=input_dim, dk_dim=dk_dim) for _ in range(n_heads)])
        self.l = nn.Linear(dk_dim * n_heads, input_dim)

    def forward(self, seq_inputs, querys=None, mask=None):
        outs = []
        for one in self.atte_layers:
            out = one(seq_inputs, querys, mask)
            outs.append(out)

        outs = torch.cat(outs, dim=-1)
        outs = self.l(outs)
        return outs


class OneHeadAttention(nn.Module):
    def __init__(self, input_dim, dk_dim):
        super(OneHeadAttention, self).__init__()
        self.dk_dim = dk_dim

        # 初始化Q, K, V:
        self.lQ = nn.Linear(input_dim, dk_dim)
        self.lK = nn.Linear(input_dim, dk_dim)

        # V的维度可以改变，只是此处方便定义全部设置成 dk_dim
        self.lV = nn.Linear(input_dim, dk_dim)

    def forward(self, seq_inputs, querys=None, mask=None):

        # seq_inputs : [batch_size, seq_lens, input_dim]
        # querys : [batch_size, seq_lens, input_dim]
        # mask : [1, seq_lens, seq_lens]

        if querys is not None:
            Q = self.lQ(querys)
        else:
            Q = self.lQ(seq_inputs)

        K = self.lK(seq_inputs)  # [ batch_size, seq_lens, dk_dim]
        V = self.lV(seq_inputs)  # [ batch_size, seq_lens, dk_dim]

        QK = torch.matmul(K, Q.permute(0, 2, 1))

        QK /= (self.dk_dim ** 0.5)

        # mask序列中0的位置变为 -1e9 遮盖此处的值
        if mask is not None:
            QK = QK.masked_fill(mask == 0, -1e9)

        alpha = torch.softmax(QK, dim=-1)

        outs = torch.matmul(alpha, V)
        return outs


# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, input_dim, ff_dim, drop_rate=0.1):
        super().__init__()
        self.l1 = nn.Linear(input_dim, ff_dim)
        self.l2 = nn.Linear(ff_dim, input_dim)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(self.drop_out(torch.relu(out1)))
        return out2


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = 10000.0 ** (torch.arange(0, input_dim, 2) / input_dim)

        # 偶数位计算sin, 奇数位计算cos
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.drop_out(x)


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self, input_dim, dk_dim, n_heads, drop_rate=0.1):
        super().__init__()
        # 初始化多头注意力
        self.attention = MultiHeadAttentionLayer(input_dim, dk_dim, n_heads)

        # 初始化注意力之后的LN
        self.a_LN = nn.LayerNorm(input_dim)

        # 初始化前馈网络
        self.ff_layer = FeedForward(input_dim, input_dim // 2)

        self.ff_layer_LN = nn.LayerNorm(input_dim)

        self.drop_out = nn.Dropout(drop_rate)

    # seq_inputs : [batch_size, seq_lens, input_dim]
    def forward(self, seq_inputs):
        # 多头注意力的输出
        outs = self.attention(seq_inputs)
        # layerNorm
        outs = self.a_LN(self.drop_out(outs) + seq_inputs)
        # 前馈神经网络
        outs_ = self.ff_layer(outs)
        outs = self.ff_layer_LN(outs + self.drop_out(outs_))
        return outs


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, dk_dim, n_heads, n_layers, drop_rate=0.1):
        """
        :param e_dim:  embedding 维度
        :param h_dim:  多头自注意力中间的 dk维度
        :param n_heads: 多头个数
        :param n_layers: N 层数
        :param drop_rate: drop out 的比例
        """
        super().__init__()
        # 两部分组成，位置编码，加上Encoder层的编码
        self.position_embedding = PositionalEncoding(input_dim)

        # 初始化N个编码层
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(input_dim, dk_dim, n_heads, drop_rate) for _ in range(n_layers)])

    def forward(self, seq_inputs):
        """
        :param seq_inputs:
        :return: [batch_size, seq_len, input_dim] 与输入的维度一致
        """
        # 先位置编码
        seq_inputs = self.position_embedding(seq_inputs)

        # 然后再送入 N 层中去编码
        for layer in self.encoder_layers:
            seq_inputs = layer(seq_inputs)

        return seq_inputs

if __name__=="__main__":
    input = torch.randn(5,3,12)
    tem = TransformerEncoder(input_dim=12, dk_dim=8, n_heads=3, n_layers=6)
    tem.forward(input)
    pass
