import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import app_predict.utils as utils
import app_predict.config as config



class SiameseCrossAttention(nn.Module):
    def __init__(self,
                 n_heads=config.cross_attention_n_heads,
                 d_q=config.cross_attention_d_q,
                 d_k=config.cross_attention_d_k,
                 d_v=config.cross_attention_d_v,
                 d_model=config.pretrain_hidden_size):
        super(SiameseCrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(self.d_model, self.d_q * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)

        # self.soft_max = nn.Softmax(dim=-1)
        self.fc = nn.Linear(n_heads * self.d_k, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self,
                last_hidden_state_a,
                attention_mask_a,
                last_hidden_state_b,
                attention_mask_b):



        # residual_a = last_hidden_state_a
        # residual_b = last_hidden_state_b

        batch_size = last_hidden_state_a.size(0)
        seq_len = attention_mask_a.size(1)

        # last_hidden_state: [batch_size, seq_len, hidden_state]
        # attention_mask: [batch_size, seq_len]
        # [batch_size, seq_len, hidden_size] -> [batch_size, n_heads, seq_len, d_q/d_k/d_v]
        q_a = self.W_Q(last_hidden_state_a).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)
        k_a = self.W_K(last_hidden_state_a).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_a = self.W_V(last_hidden_state_a).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        q_b = self.W_Q(last_hidden_state_b).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)
        k_b = self.W_K(last_hidden_state_b).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_b = self.W_V(last_hidden_state_b).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 扩展 sen 的 attention_mask
        # [batch_size, seq_len_a] -> [batch_size, 1, seq_len_a]
        attention_mask_a = attention_mask_a.data.eq(0).unsqueeze(1)
        # [batch_size, 1, seq_len_a] -> [batch_size, seq_len_a, seq_len_a]
        attention_mask_a = attention_mask_a.expand(batch_size, seq_len, seq_len)

        # [batch_size, seq_len_b] -> [batch_size, 1, seq_len_b] -> [batch_size, seq_len_b, seq_len_b]
        # 将 pad 位置改为 1，其余改为 0
        attention_mask_b = attention_mask_b.data.eq(0).unsqueeze(1)
        attention_mask_b = attention_mask_b.expand(batch_size, seq_len, seq_len)

        # [batch_size, seq_len_a, seq_len_a] -> [batch_size, n_heads, seq_len_a, seq_len_a]
        attention_mask_a = attention_mask_a.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # [batch_size, seq_len_b, seq_len_b] -> [batch_size, n_heads, seq_len_b, seq_len_b]
        attention_mask_b = attention_mask_b.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # [batch_size, n_heads, seq_len_a, d_v]
        context_a = self.cross_attention(q=q_a, k=k_b, v=v_b, mask=attention_mask_b)
        # [batch_size, n_heads, seq_len_b, d_v]
        context_b = self.cross_attention(q=q_b, k=k_a, v=v_a, mask=attention_mask_a)

        # [batch_size, seq_len, n_heads * d]
        context_a = context_a.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        context_b = context_b.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        context_a = self.fc(context_a)
        context_b = self.fc(context_b)

        context_a = self.norm(context_a)
        context_b = self.norm(context_b)

        # 不做 残差连接了
        return context_a, context_b

    def cross_attention(self, q, k, v, mask):

        # 矩阵乘法
        # 将此句子作为 q 即 Decoder
        # 使用另一个句子的 k, v 即 Encoder
        # q_a * k_b: [batch_size, n_heads, seq_len_a, d] * [batch_size, n_heads, d, seq_len_b]
        # -> [batch_size, n_heads, seq_len_a, seq_len_b]
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)

        # 将 pad 位置的评分置为0
        # 这里应该传入 Encoder 的 mask
        # mask: [batch_size, n_heads, seq_len_b, seq_len_b]
        attention_scores.masked_fill_(mask, -1e9)

        # softmax
        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        # 将注意力评分跟 Encoder 的 input 相乘，即 V，得到 context
        # attention_scores * v_b
        # [batch_size, n_heads, seq_len_a, seq_len_b] * [batch_size, n_heads, seq_len_b, d_v]
        # -> [batch_size, n_heads, seq_len_a, d_v]
        context = torch.matmul(attention_scores, v)
        return context


class MyModel(nn.Module):
    def __init__(self,
                 hidden_size=config.pretrain_hidden_size,
                 drop_out=config.similarity_drop_out,
                 num_labels=config.num_labels):
        super(MyModel, self).__init__()

        # 参数
        self.hidden_size = hidden_size
        self.dropout = drop_out
        self.num_labels = num_labels

        # 预处理模型
        # 预处理模型应该实例化一个，共同使用
        self.pretrain_model = utils.pretrain_model.to(config.device)

        # cross attention 模型
        # self.cross_attention = CrossAttention()
        # self.cross_attention_second = CrossAttention()

        self.siamese_cross_attention = SiameseCrossAttention().to(config.device)

        # 相似度计算模型
        # 应该用LayerNorm 而不是 BatchNorm
        self.similarity = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_size * 26),
            # nn.BatchNorm1d(self.hidden_size * 24),
            nn.LayerNorm(self.hidden_size * 24).to(config.device),
            # nn.LayerNorm(self.hidden_size * 8),
            # nn.BatchNorm1d(self.hidden_size * 18),

            # nn.Linear(self.hidden_size * 26, self.hidden_size),
            nn.Linear(self.hidden_size * 24, self.hidden_size).to(config.device),
            # nn.Linear(self.hidden_size * 8, self.hidden_size),
            # nn.Linear(self.hidden_size * 18, self.hidden_size),
            nn.ELU(inplace=True).to(config.device),
            # nn.BatchNorm1d(self.hidden_size),
            nn.LayerNorm(self.hidden_size).to(config.device),
            nn.Dropout(self.dropout).to(config.device),

            nn.Linear(self.hidden_size, self.hidden_size).to(config.device),
            nn.ELU(inplace=True).to(config.device),
            # nn.BatchNorm1d(self.hidden_size),
            nn.LayerNorm(self.hidden_size).to(config.device),
            nn.Dropout(self.dropout).to(config.device),

            nn.Linear(self.hidden_size, self.num_labels).to(config.device)
        )

    # 这里不用返回损失，只需要计算出结果
    def forward(self, inputs_a, inputs_b):
        # inputs 是一个字典 dict
        # dict 里面字典值是一个 tensor list 而不是一个 tensor
        # list 本身没有 size()
        # dict: {input_ids: [batch_size, seq_len],
        # token_type_ids : [batch_size, seq_len],
        # attention_mask : [batch_size, seq_len]}
        # seq_len: [CLS] + sen + [SEP]
        # 在 tokenizer指定返回tensor的时候，变成[batch_size, 1, seq_len]
        # 或许需要压缩一下
        # 这里要指定压缩，不然只有 batch_size = 1 的时候被压缩的话，维度变为 1
        inputs_a = {x: inputs_a[x].squeeze(1) for x in inputs_a}
        inputs_b = {x: inputs_b[x].squeeze(1) for x in inputs_b}

        # 编码层

        # 实参解引用
        # output -> dict{last_hidden_state: [batch_size,seq_len,hidden_state], pooler_output: [batch_size, hidden_state]
        # pooler_output 就是取[CLS]位置上的 hidden state 再过 fc 和 激活函数之后的结果
        output_a = self.pretrain_model(**inputs_a)
        output_b = self.pretrain_model(**inputs_b)

        # 交互层

        # 取出 attention_mask的位置
        attention_mask_a = inputs_a['attention_mask']
        attention_mask_b = inputs_b['attention_mask']

        # 取出编码好的句子表示
        last_hidden_state_a = output_a['last_hidden_state']
        last_hidden_state_b = output_b['last_hidden_state']

        # 取出[CLS]位置的池化表示
        cls_pool_state_a = output_a['pooler_output']
        cls_pool_state_b = output_b['pooler_output']

        # 获取注意力交互层的 context
        context_a, context_b = self.siamese_cross_attention(last_hidden_state_a=last_hidden_state_a,
                                                    attention_mask_a=attention_mask_a,
                                                    last_hidden_state_b=last_hidden_state_b,
                                                    attention_mask_b=attention_mask_b)

        # 二次注意力交互
        # context_a, context_b = self.cross_attention_second(last_hidden_state_a=context_a,
        #                                                    attention_mask_a=attention_mask_a,
        #                                                    last_hidden_state_b=context_b,
        #                                                    attention_mask_b=attention_mask_b)

        # 特征增强
        # last_hidden_state: [batch_size,seq_len,hidden_size]
        # context: [batch_size, seq_len, n_heads * d]
        # -> [batch_size, seq_len, 4 * hidden_size]
        combine_a = torch.cat([last_hidden_state_a, context_a, self.sub_and_mul(last_hidden_state_a, context_a)], -1)
        combine_b = torch.cat([last_hidden_state_b, context_b, self.sub_and_mul(last_hidden_state_b, context_b)], -1)

        # 池化
        # output: # [batch_size, 8 * hidden_size]
        pool_a = self.pooling(combine_a)
        pool_b = self.pooling(combine_b)

        # 取出 combine 中 [CLS] 的特征表示
        # [batch_size, 4 * hidden_size]
        # 这里不应该选择 combine 的，即 last_hidden_state[:,0,:]
        # 而应该选择预处理模型已经池化并激活的 特征表示
        # 这里在最后一维上拼接 cls 已经包含了 预处理模型的信息 和 交互之后的信息
        cls_a = combine_a[:, 0, :]
        cls_b = combine_b[:, 0, :]

        # 这里改动 hidden size 变化
        # [batch_size, 1 * hidden_size]
        # cls_a = cls_pool_state_a
        # cls_b = cls_pool_state_b

        # cls_cross_a = context_a[:, 0, :]
        # cls_cross_b = context_b[:, 0, :]

        # 聚合
        # 这里要不要接一下 残差，将预处理模型的结果[CLS]处的表示添加进来？
        # [batch_size, (4+4+8+8) * hidden_size] 24 * hidden_size
        # [batch_size, (1+1+8+8) * hidden_size] 18 * hidden_size
        agg_full = torch.cat([cls_a, cls_b, pool_a, pool_b], -1)

        # [batch_size, (1+1+4+4+8+8) * hidden_size] 26 * hidden_size
        # agg_full = torch.cat([cls_pool_state_a, cls_pool_state_b, cls_a, cls_b, pool_a, pool_b], -1)

        # [batch_size, (4+4) * hidden_size] 8 * hidden_size
        # agg_full = torch.cat([cls_a, cls_b], -1)

        # 输出二分类
        # [batch_size, 2]
        similarity = self.similarity(agg_full)

        # 防止值溢出
        # 便于交叉熵计算
        # 所谓交叉熵损失，就是将原本信息熵的概率值替换为概率输出值，真实值保持不变
        # 这里提前对输出的概率分布做了 log 处理
        similarity = F.log_softmax(similarity, -1)

        return similarity

    def sub_and_mul(self, a, b):

        # 对应点相乘
        mul = a * b

        # 对应点相减
        sub = a - b
        return torch.cat([sub, mul], dim=-1)

    def pooling(self, x):
        # x: [batch_size, seq_len, 4 * hidden_size]

        # 平均池化
        # [batch_size, 4 * hidden_size]
        pool_avg = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        # 最大池化
        # [batch_size, 4 * hidden_size]
        pool_max = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        # output: # [batch_size, 8 * hidden_size]
        return torch.cat([pool_avg, pool_max], -1)
