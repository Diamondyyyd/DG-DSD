import torch
import torch.nn.functional as F
from torch import nn

from ddecomposition.attention import OrdAttention, MixAttention, DualGateMixAttention



class TemporalTransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TemporalTransformerBlock, self).__init__()
        self.attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.attention(x, x, x)

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTransformerBlock, self).__init__()
        self.attention = OrdAttention(window_size, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attention(x, x, x)
        x = x.permute(0, 2, 1)

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)


class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTemporalTransformerBlock, self).__init__()
        self.time_block = TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.feature_block = SpatialTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=2 * model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        time_x = self.time_block(x)
        feature_x = self.feature_block(x)
        x = self.norm1(torch.cat([time_x, feature_x], dim=-1))

        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm2(x)


class DecompositionBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, feature_num, head_num, dropout):
        super(DecompositionBlock, self).__init__()
        # 使用双流动态门控注意力替代原版 MixAttention
        # residual=True：修复原版 residual=False 导致的信息流不对称问题
        self.mixed_attention = DualGateMixAttention(model_dim, atten_dim, head_num, dropout, True)
        self.ordinary_attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        # FFN
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        # 输出映射 [B,L,D] -> [B,L,F]
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, feature_num)

        # 🔧 改进1：可学习的低通滤波器（替代固定权重）
        self.lowpass = nn.Conv1d(in_channels=model_dim, out_channels=model_dim,
                                 kernel_size=5, padding=2, groups=model_dim, bias=True)
        # 初始化为均匀权重，但允许学习
        nn.init.constant_(self.lowpass.weight, 1.0 / 5.0)
        nn.init.zeros_(self.lowpass.bias)
        
        # 🔧 改进2：可学习的融合权重（替代固定 0.7/0.3）
        self.smooth_weight = nn.Parameter(torch.tensor(0.3))

        # 🔧 改进3：多头门控（不同特征维度不同的门控强度）
        self.gate = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, model_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, trend, time):
        """
        trend: [B, L, D]
        time:  [B, L, D]
        return:
            stable: [B, L, F]
            trend:  [B, L, D]
        """
        # -------- 稳定成分提取 --------
        stable = self.mixed_attention(trend, time, trend, time, time)  
        stable = self.ordinary_attention(stable, stable, stable)        

        residual = stable.clone()
        stable = self.activation(self.conv1(stable.permute(0, 2, 1)))   
        stable = self.dropout(self.conv2(stable).permute(0, 2, 1))      
        stable = self.norm1(stable + residual)

        # 🔧 改进：可学习的低通滤波 + 自适应融合
        smooth = self.lowpass(stable.permute(0, 2, 1)).permute(0, 2, 1) 
        smooth_w = torch.sigmoid(self.smooth_weight)  # 限制在 [0, 1]
        stable = (1 - smooth_w) * stable + smooth_w * smooth

        # 🔧 改进：多头门控机制
        gate_val = self.gate(trend)  # 已经包含 sigmoid
        trend = self.norm2(trend - gate_val * stable)                   

        # 输出稳定成分到原始特征空间
        stable = self.fc2(self.activation(self.fc1(stable)))            

        return stable, trend
