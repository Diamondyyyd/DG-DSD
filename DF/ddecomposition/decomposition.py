import torch
from torch import nn

from ddecomposition.block import SpatialTemporalTransformerBlock, TemporalTransformerBlock, DecompositionBlock
from ddecomposition.embedding import DataEmbedding, PositionEmbedding, TimeEmbedding
from ddecomposition.subtraction import OffsetSubtraction
import torch.nn.functional as F


class DataEncoder(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, block_num, head_num, dropout):
        super(DataEncoder, self).__init__()
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.encoder_blocks.append(
                SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )

    def forward(self, x):
        x = self.data_embedding(x) + self.position_embedding(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x


class TimeEncoder(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout):
        super(TimeEncoder, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)

        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.encoder_blocks.append(
                TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dp)
            )

    def forward(self, x):
        x = self.time_embed(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x





class DynamicDecomposition(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num,
                 time_num, block_num, head_num, dropout, d,
                 adaptive=False, tol=1e-3):
        super(DynamicDecomposition, self).__init__()
        self.data_encoder = DataEncoder(window_size, model_dim, ff_dim, atten_dim,
                                        feature_num, block_num, head_num, dropout)
        self.time_encoder = TimeEncoder(model_dim, ff_dim, atten_dim,
                                        time_num, block_num, head_num, dropout)

        self.decomposition_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.decomposition_blocks.append(
                DecompositionBlock(model_dim, ff_dim, atten_dim, feature_num, head_num, dp)
            )

        self.minus = OffsetSubtraction(window_size, feature_num, d, learnable=True)

        # 🔧 改进：自适应控制参数（默认开启）
        self.adaptive = adaptive
        self.tol = tol
        self.max_blocks = block_num
        
        # 🔧 改进：残差重加权（不同 block 的贡献不同）
        self.block_weights = nn.Parameter(torch.ones(block_num) / block_num)

    def forward(self, data, time):
        """
        data: [B, L, F]   原始输入
        time: [B, L, T]   时间特征
        return:
            stable: [B, L, F]  稳定成分
            trend:  [B, L, F]  趋势成分
        """
        residual = data.clone()   # 保留原始输入
        data = self.data_encoder(data)   # [B, L, D]
        time = self.time_encoder(time)   # [B, L, D]

        stable = torch.zeros_like(residual).to(data.device)  # [B, L, F]
        stable_components = []  # 存储每个 block 的输出

        prev_residual = residual
        for i, block in enumerate(self.decomposition_blocks):
            tmp_stable, data = block(data, time)   
            stable_components.append(tmp_stable)

            # 🔧 更新残差
            residual = residual - tmp_stable

            # 🔧 自适应终止：如果残差方差变化足够小，就提前停止
            if self.adaptive and i > 0:
                diff = torch.var(residual - prev_residual) / (torch.var(prev_residual) + 1e-8)
                if diff < self.tol:
                    break
                prev_residual = residual
        
        # 🔧 改进：加权融合各 block 的稳定成分
        weights = F.softmax(self.block_weights[:len(stable_components)], dim=0)
        for i, comp in enumerate(stable_components):
            stable = stable + weights[i] * comp

        # 趋势部分
        trend = self.minus(residual, stable)   # [B, L, F]
        trend = torch.mean(trend, dim=1).unsqueeze(1).repeat(1, data.shape[1], 1)  # [B, L, F]

        return stable, trend


