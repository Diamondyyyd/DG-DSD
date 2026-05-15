import numpy as np
import torch
from torch import nn


class OrdAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(OrdAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V):
        residual = Q.clone()

        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


class MixAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(MixAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Q_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q_data, Q_time, K_data, K_time, V_time):
        residual = Q_data.clone()

        Q_data = self.W_Q_data(Q_data).view(Q_data.size(0), Q_data.size(1), self.head_num, self.atten_dim)
        Q_time = self.W_Q_time(Q_time).view(Q_time.size(0), Q_time.size(1), self.head_num, self.atten_dim)
        K_data = self.W_K_data(K_data).view(K_data.size(0), K_data.size(1), self.head_num, self.atten_dim)
        K_time = self.W_K_time(K_time).view(K_time.size(0), K_time.size(1), self.head_num, self.atten_dim)
        V_time = self.W_V_time(V_time).view(V_time.size(0), V_time.size(1), self.head_num, self.atten_dim)

        Q_data, Q_time = Q_data.transpose(1, 2), Q_time.transpose(1, 2)
        K_data, K_time = K_data.transpose(1, 2), K_time.transpose(1, 2)
        V_time = V_time.transpose(1, 2)

        scores_data = torch.matmul(Q_data, K_data.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_time = torch.matmul(Q_time, K_time.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores_data + scores_time)
        context = torch.matmul(attn, V_time)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


class DualGateMixAttention(nn.Module):
    """
    双流动态门控注意力（Dual-Stream Dynamic-Gated Attention）

    针对原版 MixAttention 的两处核心结构缺陷进行改进：

    缺陷一：V 只来自时间流（W_V_time），数据流的内容信息在聚合阶段完全丢失。
    改进一：新增 W_V_data，V 由数据流和时间流双路内容融合，
            融合权重 beta 为可学习标量，让模型自动决定两流内容的相对重要性。

    缺陷二：scores_data + scores_time 固定 1:1 相加，
            任何时间步、任何注意力头的两流贡献比例完全相同，
            无法适应序列内不同位置的数据/时间主导性差异。
    改进二：逐位置、逐头的动态门控 gamma(b,h,t)，
            由每个时间步 t 处的 Q_data 和 Q_time 拼接后经轻量 MLP 生成，
            形状为 [B, H, L, 1]，在 score 维度广播，
            实现每个时间步独立决策两流 score 的混合比例。
    """
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(DualGateMixAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        # Q/K 投影（与原版结构相同）
        self.W_Q_data = nn.Linear(model_dim, atten_dim * head_num, bias=True)
        self.W_Q_time = nn.Linear(model_dim, atten_dim * head_num, bias=True)
        self.W_K_data = nn.Linear(model_dim, atten_dim * head_num, bias=True)
        self.W_K_time = nn.Linear(model_dim, atten_dim * head_num, bias=True)

        # 改进一：新增数据流 V 投影
        self.W_V_data = nn.Linear(model_dim, atten_dim * head_num, bias=True)
        self.W_V_time = nn.Linear(model_dim, atten_dim * head_num, bias=True)
        # 可学习 V 融合权重：V = sigmoid(beta)*V_data + (1-sigmoid(beta))*V_time
        self.beta = nn.Parameter(torch.tensor(0.0))  # 初始 sigmoid(0)=0.5，两流等权

        # 改进二：逐位置逐头门控网络
        # 输入：每个时间步 t 处拼接的 [Q_d(b,h,t,:), Q_t(b,h,t,:)] → 2*atten_dim
        # 输出：gamma(b,h,t,1)，在 score 列维度广播
        # 轻量设计：单隐层，隐层维度 = atten_dim // 2，避免参数过多
        self.gate_net = nn.Sequential(
            nn.Linear(2 * atten_dim, atten_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(atten_dim // 2, 1, bias=True),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(atten_dim * head_num, model_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q_data, Q_time, K_data, K_time, V_time):
        """
        Q_data, K_data : 数据流表示  [B, L, model_dim]
        Q_time, K_time, V_time : 时间流表示  [B, L, model_dim]
        返回: [B, L, model_dim]
        """
        B, L, _ = Q_data.shape
        residual = Q_data

        # ── Q/K 投影并分头 ──────────────────────────────────────────
        # [B, L, D] → [B, H, L, d_k]
        Q_d = self.W_Q_data(Q_data).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        Q_t = self.W_Q_time(Q_time).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        K_d = self.W_K_data(K_data).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        K_t = self.W_K_time(K_time).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)

        # ── 改进一：双流 V 融合 ─────────────────────────────────────
        # V_data 来自数据流自身内容，V_time 来自时间流
        V_d = self.W_V_data(Q_data).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        V_t = self.W_V_time(V_time).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        beta = torch.sigmoid(self.beta)          # 标量，在 [0,1]
        V = beta * V_d + (1.0 - beta) * V_t     # [B, H, L, d_k]

        # ── 改进二：逐位置逐头动态门控 gamma ────────────────────────
        # Q_d, Q_t: [B, H, L, d_k]
        # 拼接两流在每个位置的查询向量 → [B, H, L, 2*d_k]
        gate_input = torch.cat([Q_d, Q_t], dim=-1)      # [B, H, L, 2*d_k]
        # 经轻量 MLP 生成逐位置门控权重 → [B, H, L, 1]
        gamma = self.gate_net(gate_input)                # [B, H, L, 1]
        # gamma 广播到 [B, H, L, L]：query 维度的每个位置有自己的混合比例

        # ── Score 动态加权融合（替代固定 1:1 相加）──────────────────
        scale = np.sqrt(self.atten_dim)
        scores_data = torch.matmul(Q_d, K_d.transpose(-1, -2)) / scale  # [B, H, L, L]
        scores_time = torch.matmul(Q_t, K_t.transpose(-1, -2)) / scale  # [B, H, L, L]
        # gamma 作用在 query 位置（dim=-2），对 key 维度广播
        scores = gamma * scores_data + (1.0 - gamma) * scores_time       # [B, H, L, L]

        attn = torch.softmax(scores, dim=-1)             # [B, H, L, L]
        context = torch.matmul(self.dropout(attn), V)    # [B, H, L, d_k]

        # ── 合并多头，输出投影 ───────────────────────────────────────
        context = context.transpose(1, 2).reshape(B, L, -1)  # [B, L, H*d_k]
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)
