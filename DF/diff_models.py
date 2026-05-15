# diff_models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class _SafeOps:
    @staticmethod
    def clamp_nan(x, minv=-1e4, maxv=1e4):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return torch.clamp(x, min=minv, max=maxv)


class LinearSelfAttention(nn.Module):
    def __init__(self, channels, nheads, dropout=0.0):
        super().__init__()
        assert channels % nheads == 0
        self.channels = channels
        self.nheads = nheads
        self.head_dim = channels // nheads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        # init
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _phi(x):

        return F.elu(x) + 1.0

    def forward(self, x):
        """
        x: (B, S, C)
        returns: (B, S, C)
        """
        B, S, C = x.shape
        x = _SafeOps.clamp_nan(x)

        q = self.q_proj(x).view(B, S, self.nheads, self.head_dim).transpose(1, 2)  
        k = self.k_proj(x).view(B, S, self.nheads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nheads, self.head_dim).transpose(1, 2)

        qf = self._phi(q)  
        kf = self._phi(k)  

    
        kv = torch.einsum('bhsd,bhse->bhde', kf, v)  
        kf_sum = kf.sum(dim=2)  

   
        out_raw = torch.einsum('bhsd,bhde->bhse', qf, kv)

       
        z = torch.einsum('bhsd,bhd->bhs', qf, kf_sum).clamp(min=1e-6)  

        out = out_raw / z.unsqueeze(-1)  
        out = out.transpose(1, 2).contiguous().view(B, S, C)  
        out = self.out_proj(self.dropout(out))
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out



class FeedForward(nn.Module):
    def __init__(self, channels, mult=4, dropout=0.0):
        super().__init__()
        inner = channels * mult
        self.net = nn.Sequential(
            nn.Linear(channels, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, channels),
            nn.Dropout(dropout)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = _SafeOps.clamp_nan(x)
        return self.net(x)


class EfficientTransformerBlock(nn.Module):
    def __init__(self, channels, nheads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = LinearSelfAttention(channels, nheads, dropout=dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels, mult=4, dropout=dropout)

    def forward(self, x):
        x = _SafeOps.clamp_nan(x)
        h = self.norm1(x)
        a = self.attn(h)
        x = x + a
        x = x + self.ff(self.norm2(x))
        x = _SafeOps.clamp_nan(x)
        return x


class EfficientTimeTransformer(nn.Module):
    """时间轴 Transformer：接收 (B*K, C, L) 并返回 (B*K, C, L)"""
    def __init__(self, channels, nheads, depth=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([EfficientTransformerBlock(channels, nheads, dropout=dropout) for _ in range(depth)])

    def forward(self, x):  
        BK, C, L = x.shape
        if L == 1:
            return x
        x = x.transpose(1, 2).contiguous()  # (B*K, L, C)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).contiguous()  # (B*K, C, L)
        return x

class EfficientFeatureTransformer(nn.Module):
    """特征轴 Transformer：接收 (B*L, C, K) 并返回 (B*L, C, K)"""
    def __init__(self, channels, nheads, depth=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([EfficientTransformerBlock(channels, nheads, dropout=dropout) for _ in range(depth)])

    def forward(self, x):  # x: (B*L, C, K)
        BL, C, K = x.shape
        if K == 1:
            return x
        x = x.transpose(1, 2).contiguous()  
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).contiguous()  # (B*L, C, K)
        return x



class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积：Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        # Depthwise卷积：每个通道独立卷积
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels  # 关键：groups=in_channels
        )
        
        # Pointwise卷积：1x1卷积混合通道
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 初始化
        nn.init.kaiming_normal_(self.depthwise.weight)
        nn.init.kaiming_normal_(self.pointwise.weight)
        if self.depthwise.bias is not None:
            nn.init.zeros_(self.depthwise.bias)
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvFFN(nn.Module):
    """卷积前馈网络"""
    def __init__(self, channels, expansion=4, dropout=0.1):
        super().__init__()
        hidden_channels = channels * expansion
        self.conv1 = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class LocalDMTB(nn.Module):
    """
    局部 DMTB（单路深度可分离卷积）
    
    用于局部分支，固定 dilation=1，单路 DWSConv + ConvFFN + 残差。
    职责单一：只建模邻域局部细节，参数量约为三路 DMTB 的 1/3。
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # 单路深度可分离卷积
        self.dwconv = DepthwiseSeparableConv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout
        )
        # 归一化
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        # 卷积前馈网络
        self.ffn = ConvFFN(channels, expansion=4, dropout=dropout)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, C, L)
        returns: (B, C, L)
        """
        # 子层一：单路 DWSConv + 残差
        residual = x
        x = self.norm1(x)
        x = self.dwconv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = residual + x
        # 子层二：ConvFFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class DMTB(nn.Module):
    """
    Dilated ModernTCN Block (DMTB) — 三路并行版，用于全局分支
    
    包含：
    1. 三路并行深度可分离卷积（dilation, dilation*2, dilation*4）
    2. 融合卷积
    3. 卷积前馈网络（ConvFFN）
    4. 残差连接
    5. 归一化
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        # 三路并行深度可分离卷积，覆盖三个相邻尺度
        self.dwconv_list = nn.ModuleList()
        dilations = [dilation, dilation * 2, dilation * 4]
        for d in dilations:
            self.dwconv_list.append(
                DepthwiseSeparableConv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout
                )
            )
        
        # 融合卷积：3C -> C
        self.fusion_conv = nn.Conv1d(channels * len(dilations), channels, kernel_size=1)
        
        # 归一化
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # 卷积前馈网络
        self.ffn = ConvFFN(channels, expansion=4, dropout=dropout)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, C, L)
        returns: (B, C, L)
        """
        # 子层一：三路并行 DWSConv + 融合 + 残差
        residual = x
        x = self.norm1(x)
        dwconv_outputs = [dwconv(x) for dwconv in self.dwconv_list]
        x = torch.cat(dwconv_outputs, dim=1)  # (B, 3C, L)
        x = self.fusion_conv(x)               # (B, C, L)
        x = self.activation(x)
        x = self.dropout(x)
        x = residual + x
        # 子层二：ConvFFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class ModernTCNEncoder(nn.Module):
    """
    ModernTCN编码器（改进版：双路设计）
    
    🔥 核心改进：明确分离局部和全局建模
    
    架构设计：
    1. 局部分支（Tcn_Local）：固定膨胀率 d=1，捕捉局部细节和短期依赖
    2. 全局分支（Tcn_Global）：递增膨胀率 d=1,2,4,8...，捕捉全局依赖和长期模式
    3. 门控融合：自适应加权融合两个分支
    4. Transformer增强：捕捉长距离时间依赖
    
    为什么这样改是必要的：
    - 旧设计：每层DMTB内部混合多膨胀率（d=1,2,4），信息混乱，职责不清
    - 新设计：明确分离，对应DTAAD的成功经验，架构更清晰
    - 效果：局部和全局信息分别建模，融合时更有针对性
    """
    def __init__(self, channels, num_layers=3, kernel_size=3, 
                 diffusion_embedding_dim=128, use_transformer=True, 
                 nheads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.use_transformer = use_transformer
        
        # 扩散步嵌入投影
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim, channels)
        
        # ============ 局部分支：固定 dilation=1，单路 LocalDMTB ============
        # 作用：捕捉局部细节、短期波动、邻域信息
        # 使用单路 DWSConv，参数量约为三路 DMTB 的 1/3，职责单一
        self.local_layers = nn.ModuleList()
        for i in range(num_layers):
            self.local_layers.append(
                LocalDMTB(channels, kernel_size=kernel_size, dilation=1, dropout=dropout)
            )
        
        # ============ 全局分支：递增膨胀率 d=2,4,8,...，三路 DMTB ============
        # 作用：捕捉全局依赖、长期模式、多尺度信息
        # 从 dilation=2 开始，避免与局部分支（dilation=1）在第0层完全重叠
        self.global_layers = nn.ModuleList()
        for i in range(num_layers):
            base_dilation = 2 ** (i + 1)  # 2, 4, 8, 16... 与局部分支真正分离
            self.global_layers.append(
                DMTB(channels, kernel_size=kernel_size, dilation=base_dilation, dropout=dropout)
            )
        
        # ============ 门控融合：自适应加权两个分支 ============
        # 让模型学习在不同位置/时间选择局部或全局信息
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Transformer增强（可选）
        if use_transformer:
            self.transformer = EfficientTimeTransformer(channels, nheads, depth=1, dropout=dropout)
        
        # 输出归一化
        self.output_norm = nn.GroupNorm(8, channels)
    
    def forward(self, x, diffusion_emb=None):
        """
        x: (B*K, C, L)
        diffusion_emb: (B, emb_dim) - 扩散步嵌入
        returns: (B*K, C, L)
        
        处理流程：
        1. 注入扩散步嵌入
        2. 局部分支处理
        3. 全局分支处理
        4. 门控融合
        5. Transformer增强
        6. 输出归一化
        """
        # 注入扩散步嵌入
        if diffusion_emb is not None:
            BK, C, L = x.shape
            # 投影扩散嵌入: (B, emb_dim) -> (B, C)
            diff_proj = self.diffusion_proj(diffusion_emb)  # (B, C)
            
            # 扩展维度: (B, C) -> (B, C, L)
            diff_proj = diff_proj.unsqueeze(-1).expand(-1, -1, L)  # (B, C, L)
            
            # 重复K次（对应K个特征）
            K = BK // diffusion_emb.shape[0]
            diff_proj = diff_proj.repeat_interleave(K, dim=0)  # (B*K, C, L)
            
            x = x + diff_proj
        
        # ============ 局部分支：固定膨胀率处理 ============
        local_x = x
        for layer in self.local_layers:
            local_x = layer(local_x)
        
        # ============ 全局分支：递增膨胀率处理 ============
        global_x = x
        for layer in self.global_layers:
            global_x = layer(global_x)
        
        # ============ 门控融合：自适应加权 ============
        # 拼接两个分支的输出
        combined = torch.cat([local_x, global_x], dim=1)  # (B*K, 2*C, L)
        
        # 计算门控权重（局部分支的权重）
        gate = self.fusion_gate(combined)  # (B*K, C, L)
        
        # 加权融合：gate * local + (1-gate) * global
        x = gate * local_x + (1 - gate) * global_x
        
        # ============ Transformer增强 ============
        if self.use_transformer:
            x = self.transformer(x)
        
        # 输出归一化
        x = self.output_norm(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,
                 time_depth=1, feat_depth=1, use_mstc=False):
        super().__init__()
        self.channels = channels

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.strategy_projection  = nn.Linear(diffusion_embedding_dim, channels)

        # projections
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection  = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection= Conv1d_with_init(2 * channels, 2 * channels, 1)

        # efficient transformers
        self.time_layer = EfficientTimeTransformer(channels, nheads, depth=time_depth, dropout=0.1)
        self.feature_layer = EfficientFeatureTransformer(channels, nheads, depth=feat_depth, dropout=0.1)

        self.cond_gate = nn.Sequential(nn.Linear(2 * channels, 2 * channels, bias=False), nn.Sigmoid())
        self.fusion_layernorm = nn.LayerNorm(2 * channels)
        self.skip_norm = nn.GroupNorm(1, channels)
        
        # 🔥 改进：自适应残差缩放（替代固定的 1/sqrt(2)）
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        
        # 跨轴交互（时间-特征双向耦合）：输入为 temporal 和 feature 的拼接
        self.cross_axis_fusion = nn.Sequential(
            nn.Conv1d(channels * 2, channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    @staticmethod
    def _reshape_for_time_attention(y, B, K, L, C):
        """
        🔥 改进：简化reshape操作
        
        将 (B, C, K*L) 转换为 (B*K, C, L)
        用于时间轴Transformer处理
        
        Args:
            y: (B, C, K*L) - 输入
            B, K, L, C: 维度参数
        Returns:
            y_: (B*K, C, L) - 输出
        """
        # (B, C, K*L) -> (B, C, K, L) -> (B, K, C, L) -> (B*K, C, L)
        y_ = y.reshape(B, C, K, L)  # (B, C, K, L)
        y_ = y_.permute(0, 2, 1, 3)  # (B, K, C, L)
        y_ = y_.reshape(B * K, C, L)  # (B*K, C, L)
        return y_
    
    @staticmethod
    def _reshape_back_from_time(out, B, K, L, C):
        """
        🔥 改进：简化reshape操作
        
        将 (B*K, C, L) 转换回 (B, C, K*L)
        用于时间轴Transformer处理后恢复
        
        Args:
            out: (B*K, C, L) - 输入
            B, K, L, C: 维度参数
        Returns:
            out: (B, C, K*L) - 输出
        """
        # (B*K, C, L) -> (B, K, C, L) -> (B, C, K, L) -> (B, C, K*L)
        out = out.reshape(B, K, C, L)  # (B, K, C, L)
        out = out.permute(0, 2, 1, 3)  # (B, C, K, L)
        out = out.reshape(B, C, K * L)  # (B, C, K*L)
        return out
    
    @staticmethod
    def _reshape_for_feature_attention(y, B, K, L, C):
        """
        🔥 改进：简化reshape操作
        
        将 (B, C, K*L) 转换为 (B*L, C, K)
        用于特征轴Transformer处理
        
        Args:
            y: (B, C, K*L) - 输入
            B, K, L, C: 维度参数
        Returns:
            y_: (B*L, C, K) - 输出
        """
        # (B, C, K*L) -> (B, C, K, L) -> (B, L, C, K) -> (B*L, C, K)
        y_ = y.reshape(B, C, K, L)  # (B, C, K, L)
        y_ = y_.permute(0, 3, 1, 2)  # (B, L, C, K)
        y_ = y_.reshape(B * L, C, K)  # (B*L, C, K)
        return y_
    
    @staticmethod
    def _reshape_back_from_feature(out, B, K, L, C):
        """
        🔥 改进：简化reshape操作
        
        将 (B*L, C, K) 转换回 (B, C, K*L)
        用于特征轴Transformer处理后恢复
        
        Args:
            out: (B*L, C, K) - 输入
            B, K, L, C: 维度参数
        Returns:
            out: (B, C, K*L) - 输出
        """
        # (B*L, C, K) -> (B, L, C, K) -> (B, C, K, L) -> (B, C, K*L)
        out = out.reshape(B, L, C, K)  # (B, L, C, K)
        out = out.permute(0, 2, 3, 1)  # (B, C, K, L)
        out = out.reshape(B, C, K * L)  # (B, C, K*L)
        return out

    def forward_time(self, y, base_shape):
        """
        时间轴Transformer处理
        
        🔥 改进：使用辅助函数简化reshape操作
        """
        B, channel, K, L = base_shape
        if L == 1:
            return y
        
        # 转换为时间轴格式: (B, C, K*L) -> (B*K, C, L)
        y_ = self._reshape_for_time_attention(y, B, K, L, channel)
        
        # 时间轴Transformer处理
        out = self.time_layer(y_)  
        
        # 转换回原始格式: (B*K, C, L) -> (B, C, K*L)
        out = self._reshape_back_from_time(out, B, K, L, channel)
        return out

    def forward_feature(self, y, base_shape):
        """
        特征轴Transformer处理
        
        🔥 改进：使用辅助函数简化reshape操作
        """
        B, channel, K, L = base_shape
        if K == 1:
            return y
        
        # 转换为特征轴格式: (B, C, K*L) -> (B*L, C, K)
        y_ = self._reshape_for_feature_attention(y, B, K, L, channel)
        
        # 特征轴Transformer处理
        out = self.feature_layer(y_)
        
        # 转换回原始格式: (B*L, C, K) -> (B, C, K*L)
        out = self._reshape_back_from_feature(out, B, K, L, channel)
        return out

    def forward(self, x, cond_info, diffusion_emb, strategy_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x_flat = x.reshape(B, channel, K * L)

        diffusion_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        strategy_proj  = self.strategy_projection(strategy_emb).unsqueeze(-1)

        y = x_flat + diffusion_proj + strategy_proj
        
        # 条件信息投影（只计算一次，后续复用）
        cond_flat = cond_info.reshape(B, cond_info.size(1), K * L)
        cond_proj = self.cond_projection(cond_flat)  # (B, 2*C, K*L)

        # 早期条件融合：取前 channel 维做弱融合（在 Transformer 之前注入）
        cond_early_signal = cond_proj[:, :channel, :]  # (B, C, K*L)
        y = y + cond_early_signal * 0.1  # 弱融合，避免压制原始信号

        temporal_out = self.forward_time(y, base_shape)    
        feature_out  = self.forward_feature(temporal_out, base_shape)  
        
        # 跨轴交互融合（时间-特征双向耦合）：拼接两路信息计算门控
        cross_input = torch.cat([temporal_out, feature_out], dim=1)  # (B, 2*C, K*L)
        cross_weight = self.cross_axis_fusion(cross_input)             # (B, C, K*L)
        feature_out = feature_out * cross_weight + temporal_out * (1.0 - cross_weight)

        # mid projection
        y_mid = self.mid_projection(feature_out)

        # 复用已计算的 cond_proj，不再重复调用 cond_projection
        cond = cond_proj  

        y_pooled = F.adaptive_avg_pool1d(y_mid, 1).squeeze(-1)     
        cond_pooled = F.adaptive_avg_pool1d(cond, 1).squeeze(-1)
        fusion_input = self.fusion_layernorm(y_pooled + cond_pooled)
        fusion_gate = self.cond_gate(fusion_input).unsqueeze(-1)  
        y_mid = y_mid + fusion_gate * cond

        # output projection
        y_out = self.output_projection(y_mid)  
        residual, skip = torch.chunk(y_out, 2, dim=1)  

        residual = residual.reshape(B, channel, K, L)
        skip = skip.reshape(B, channel, K, L)

        skip_normalized = self.skip_norm(skip)
        
        # 🔥 改进：使用自适应缩放而非固定的 1/sqrt(2)
        # 缩放因子在 [0.5, 1.5] 范围内，由模型学习
        scale = torch.clamp(self.residual_scale, min=0.5, max=1.5)
        out = (x + residual) * scale

        return out, skip_normalized



class StableComponentProjection(nn.Module):
    """
    🔥 改进：稳定分量的可学习投影层
    
    将 (B, 1, K, L) 投影到 (B, channels, K, L)
    使用可学习的参数而非随机权重
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 使用 Conv2d 进行可学习投影
        self.proj = nn.Conv2d(1, channels, kernel_size=1, bias=True)
        # 初始化权重
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, K, L) - 稳定分量
        Returns:
            out: (B, channels, K, L) - 投影后的特征
        """
        return self.proj(x)


class VariantComponentProjection(nn.Module):
    """
    🔥 改进：变化分量的可学习投影层
    
    将 (B*K, 1, L) 投影到 (B*K, channels, L)
    使用可学习的参数而非随机权重
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 使用 Conv1d 进行可学习投影
        self.proj = nn.Conv1d(1, channels, kernel_size=1, bias=True)
        # 初始化权重
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B*K, 1, L) - 变化分量
        Returns:
            out: (B*K, channels, L) - 投影后的特征
        """
        return self.proj(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1).float()
        frequencies = 10.0 ** (torch.arange(dim).float() / max(1, dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table  # (T, dim*2)


class diff_CSDI(nn.Module):
    """
    频率增强的扩散去噪网络
    
    核心改进：
    1. 在输入投影后，对信号进行频域分解（STFT）
    2. 分离稳定频率成分（时间不变）和变化频率成分（时间变化）
    3. 使用双编码器分别建模：
       - 稳定编码器（Transformer）：建模周期性结构
       - 变化编码器（ModernTCN）：建模漂移和变化
    4. 门控融合两个分支
    5. 输入到原有的ResidualBlock进行去噪
    """
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        
        # 🔥 是否启用频率增强（可通过配置控制）
        self.use_freq_enhancement = config.get("use_freq_enhancement", True)
        
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.strategy_embedding = nn.Embedding(2, config['diffusion_embedding_dim'])

        # ============ 🔥 频率增强模块 ============
        if self.use_freq_enhancement:
            # STFT参数
            self.n_fft = config.get("n_fft", 32)
            self.hop_length = config.get("hop_length", 8)
            
            # 🔥 改进：预先创建STFT window，避免每次forward重复创建
            self.register_buffer(
                "stft_window",
                torch.hann_window(self.n_fft),
                persistent=False
            )
            
            # 🔥 改进：可学习的频率选择（基于稳定性指标）
            # MLP输入3维特征：[mu_norm, cv_norm, stability_score]
            self.freq_selection_mlp = nn.Sequential(
                nn.Linear(3, 16),  # 输入3维而非1维
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
            # 可学习的温度参数
            self.freq_gate_temperature = nn.Parameter(torch.tensor(1.0))
            # 双指标权重（稳定性 vs 重要性）
            self.stability_weight = config.get("stability_weight", 0.7)  # α参数
            
            # 🔥 改进1：稳定分量的可学习投影层（替代随机权重）
            self.stable_proj = StableComponentProjection(self.channels)
            
            # 变化分量的可学习投影层
            self.variant_proj = VariantComponentProjection(self.channels)
            # 变化分支条件注入：将 cond_info 弱注入到变化编码器输入
            self.variant_cond_proj = nn.Conv1d(config["side_dim"], self.channels, kernel_size=1)
            
            # 🔥 稳定频率编码器：直接复用主干 residual_layers，消除冗余参数
            # 稳定分支投影后送入主干网络编码，skip 输出与变化分支融合后直接作为最终输出
            # 不再单独定义 stable_encoder，节省参数量并避免两套 ResidualBlock 的职责混乱
            
            # 🔥 变化频率编码器（完整版ModernTCN）
            num_tcn_layers = config.get("num_freq_tcn_layers", 3)
            use_transformer = config.get("use_tcn_transformer", True)  # 是否使用Transformer增强
            self.variant_encoder = ModernTCNEncoder(
                channels=self.channels,
                num_layers=num_tcn_layers,
                kernel_size=3,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                use_transformer=use_transformer,
                nheads=config["nheads"],
                dropout=0.1
            )
            
            # 门控融合
            fusion_type = config.get("freq_fusion_type", "gated")
            if fusion_type == "gated":
                # 输入：h_stable(C) + h_variant(C) + diffusion_bias(C) = 3C
                self.diff_to_gate = nn.Linear(config["diffusion_embedding_dim"], self.channels)
                self.fusion_gate = nn.Sequential(
                    nn.Conv2d(self.channels * 3, self.channels, kernel_size=1),
                    nn.Sigmoid()
                )
            elif fusion_type == "attention":
                self.fusion_query = nn.Conv2d(self.channels, self.channels, kernel_size=1)
                self.fusion_key = nn.Conv2d(self.channels, self.channels, kernel_size=1)
                self.fusion_value = nn.Conv2d(self.channels, self.channels, kernel_size=1)
            self.fusion_type = fusion_type
            self.fusion_norm = nn.GroupNorm(8, self.channels)
            
            # 频率增强后的输出投影（将融合特征对齐到输出空间）
            # 稳定分支 skip 加权求和后与变化分支融合，直接送入输出投影
            self.freq_to_output = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        
        # ============ 标准输入投影（频率增强关闭时使用） ============
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        
        # ============ 输出投影 ============
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # ============ ResidualBlock ============
        layers = config["layers"]
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"]
            )
            for _ in range(layers)
        ])

        self.register_parameter("skip_weights", nn.Parameter(torch.zeros(layers)))
    
    def frequency_decomposition(self, x):
        """
        频域分解：使用STFT分离稳定和变化频率成分
        
        🔥 改进版本：
        1. 使用频率稳定性指标（CV）而非幅值
        2. 幅值Mask过滤噪声频段
        3. 双指标融合（稳定性 + 重要性）
        4. 可学习门控 + 温度控制
        
        Args:
            x: (B, K, L) - 时域信号
        Returns:
            x_stable: (B, K, L) - 稳定频率成分
            x_variant: (B, K, L) - 变化频率成分
        """
        B, K, L = x.shape
        
        # 重塑为 (B*K, L)
        x_flat = x.reshape(B * K, L)
        
        # STFT
        stft_result = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window,
            return_complex=True,
            center=True,
            normalized=False
        )
        # stft_result: (B*K, freq_bins, time_frames)
        
        # 🔥 改进1：计算频率稳定性指标
        magnitude = torch.abs(stft_result)  # (B*K, freq_bins, time_frames)
        
        # 计算每个频率的统计量
        mu_f = magnitude.mean(dim=-1)  # 均值 (B*K, freq_bins)
        sigma_f = magnitude.std(dim=-1)  # 标准差 (B*K, freq_bins)
        
        # 🔥 改进1.1：Robust CV计算（避免低幅值频率的不稳定性）
        # 使用 clamp 防止除以极小值，避免数值爆炸
        eps = 1e-6
        cv_f_robust = sigma_f / (torch.clamp(mu_f, min=eps))
        cv_log = torch.log1p(cv_f_robust)
        
        # 🔥 改进2：双阈值过滤（自适应且鲁棒）
        # 使用能量阈值 + 相对阈值的组合
        if hasattr(torch, 'quantile'):
            threshold_energy = torch.quantile(mu_f, 0.15, dim=-1, keepdim=True)
        else:
            k = max(1, int(mu_f.shape[-1] * 0.15))
            threshold_energy = torch.kthvalue(mu_f, k, dim=-1, keepdim=True)[0]
        
        # 相对阈值：相对于该样本的最大能量
        max_energy = mu_f.max(dim=-1, keepdim=True)[0]
        threshold_relative = max_energy * 0.05  # 5% 相对阈值
        
        # 综合阈值：取两者的最大值（更保守）
        threshold = torch.max(threshold_energy, threshold_relative)
        valid_mask = (mu_f > threshold).float()  # (B*K, freq_bins)
        
        # 🔥 改进3：双指标稳定性得分
        # 归一化Log-CV（低CV = 高稳定性）
        cv_log_min = cv_log.min(dim=-1, keepdim=True)[0]
        cv_log_max = cv_log.max(dim=-1, keepdim=True)[0]
        cv_norm = (cv_log - cv_log_min) / (cv_log_max - cv_log_min + 1e-8)
        
        # 归一化幅值（高幅值 = 高重要性）
        mu_min = mu_f.min(dim=-1, keepdim=True)[0]
        mu_max = mu_f.max(dim=-1, keepdim=True)[0]
        mu_norm = (mu_f - mu_min) / (mu_max - mu_min + 1e-8)
        
        # 综合稳定性得分 = α * (低CV) + (1-α) * (高幅值)
        stability_score = (
            self.stability_weight * (1.0 - cv_norm) + 
            (1.0 - self.stability_weight) * mu_norm
        )
        
        # 应用有效性Mask（无效频率得分设为0）
        stability_score = stability_score * valid_mask  # (B*K, freq_bins)
        
        # 🔥 改进4：多维特征输入MLP（避免信息瓶颈）
        # 将原始特征（mu_norm, cv_norm）和综合得分都输入MLP
        # 让模型自己学习复杂的频率选择策略
        freq_features = torch.stack([
            mu_norm,          # 幅值（重要性）
            cv_norm,          # 变异系数（稳定性）
            stability_score   # 综合得分
        ], dim=-1)  # (B*K, freq_bins, 3)
        
        # 通过MLP学习频率选择策略
        freq_gates = self.freq_selection_mlp(freq_features)  # (B*K, freq_bins, 1)
        freq_gates = freq_gates.squeeze(-1)  # (B*K, freq_bins)
        
        # 温度控制的Sigmoid（温度越小，决策越锐利）
        temperature = self.freq_gate_temperature.clamp(min=0.1, max=2.0)
        freq_gates = torch.sigmoid(freq_gates / temperature)  # (B*K, freq_bins)
        
        # 🔥 改进：使用软选择而非硬掩码，改善梯度流
        # 硬掩码：stable_mask = freq_gates.unsqueeze(-1)
        # 软选择：使用加权组合，保留两个分支的信息
        stable_weight = freq_gates.unsqueeze(-1)  # (B*K, freq_bins, 1)
        variant_weight = 1.0 - stable_weight
        
        # 分离稳定和变化频率（软选择）
        stft_stable = stft_result * stable_weight
        stft_variant = stft_result * variant_weight
        
        # ISTFT回到时域
        x_stable_flat = torch.istft(
            stft_stable,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window,
            center=True,
            normalized=False,
            length=L
        )
        
        x_variant_flat = torch.istft(
            stft_variant,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window,
            center=True,
            normalized=False,
            length=L
        )
        
        # 重塑回 (B, K, L)
        x_stable = x_stable_flat.reshape(B, K, L)
        x_variant = x_variant_flat.reshape(B, K, L)
        
        return x_stable, x_variant
    
    def encode_stable_component(self, x_stable, cond_info, diffusion_emb, strategy_emb):
        """
        稳定频率编码器：直接复用主干 residual_layers 编码稳定频率成分
        
        稳定分支始终全力输出，不受置信度压制，确保稳定成分在任何扩散步都能
        得到充分训练和有效表达。
        skip 输出加权求和后与变化分支融合，作为最终去噪结果，不再重复走一遍残差块。
        
        Args:
            x_stable: (B, K, L) - 稳定频率成分
            cond_info: (B, cond_dim, K, L) - 外部条件信息
            diffusion_emb: (B, emb_dim) - 扩散时间步嵌入
            strategy_emb: (B, emb_dim) - 策略嵌入
        Returns:
            skip_agg: (B, channels, K, L) - 主干 skip 加权求和结果
        """
        B, K, L = x_stable.shape
        
        # 投影到高维空间: (B, K, L) -> (B, channels, K, L)
        x = x_stable.unsqueeze(1)  # (B, 1, K, L)
        x = self.stable_proj(x)    # (B, channels, K, L)
        x = F.relu(x)
        
        # 🔥 直接复用主干 residual_layers，收集 skip 输出
        skip_sum = None
        cur = x
        weights = torch.softmax(self.skip_weights.to(x.device), dim=0)
        for i, layer in enumerate(self.residual_layers):
            cur, skip_conn = layer(cur, cond_info, diffusion_emb, strategy_emb)
            skip_conn = torch.clamp(skip_conn, -10.0, 10.0)
            w = weights[i]
            if skip_sum is None:
                skip_sum = skip_conn * w
            else:
                skip_sum = skip_sum + skip_conn * w
        
        skip_agg = skip_sum if skip_sum is not None else torch.zeros_like(cur)
        return skip_agg
    
    def encode_variant_component(self, x_variant, diffusion_emb, cond_info):
        """
        变化频率编码器：使用完整版ModernTCN建模时间变化成分
        
        注入 cond_info（已观测值），修复稳定/变化分支的信息不对称问题。
        变化成分（局部偏差/异常）是相对于观测定义的，必须感知条件信息。
        
        Args:
            x_variant: (B, K, L) - 变化频率成分
            diffusion_emb: (B, emb_dim) - 扩散时间步嵌入
            cond_info: (B, cond_dim, K, L) - 外部条件信息（已观测值）
        Returns:
            h_variant: (B, channels, K, L) - 变化成分嵌入
        """
        B, K, L = x_variant.shape
        
        # 投影到高维空间: (B*K, 1, L) -> (B*K, channels, L)
        x = x_variant.reshape(B * K, 1, L)
        x = self.variant_proj(x)   # (B*K, channels, L)
        x = F.relu(x)
        
        # 条件信息弱注入：cond_info -> (B*K, cond_dim, L) -> (B*K, channels, L)
        # 弱注入（×0.1）避免条件信息压制变化分量本身的信号
        cond_flat = cond_info.reshape(B * K, cond_info.shape[1], L)  # (B*K, cond_dim, L)
        cond_feat = self.variant_cond_proj(cond_flat)                 # (B*K, channels, L)
        x = x + cond_feat * 0.1
        
        # 通过ModernTCN编码（注入扩散步嵌入）
        x = self.variant_encoder(x, diffusion_emb)
        
        # 重塑为 (B, K, channels, L) -> (B, channels, K, L)
        x = x.reshape(B, K, self.channels, L)
        x = x.permute(0, 2, 1, 3)
        
        return x
    
    def fuse_dual_branch(self, h_stable, h_variant, diffusion_emb):
        """
        双分支融合：融合稳定和变化频率成分
        
        Args:
            h_stable: (B, channels, K, L) - 稳定成分嵌入
            h_variant: (B, channels, K, L) - 变化成分嵌入
            diffusion_emb: (B, emb_dim) - 扩散步嵌入（显式注入门控，实现一阶条件建模）
        Returns:
            h_fused: (B, channels, K, L) - 融合后的嵌入
        """
        if self.fusion_type == "gated":
            # 门控融合：显式注入 diffusion_emb，实现一阶条件建模
            # gate = f(h_stable, h_variant, t)，避免从特征中二阶反推扩散步
            # diff_bias: (B, emb_dim) -> (B, C) -> (B, C, 1, 1) -> 广播到 (B, C, K, L)
            diff_bias = self.diff_to_gate(diffusion_emb)          # (B_or_1, C)
            diff_bias = diff_bias.unsqueeze(-1).unsqueeze(-1)      # (B_or_1, C, 1, 1)
            # 以 h_variant 的形状为基准，对齐所有维度（包括 batch 和空间维度）
            # diffusion_step 传入时可能是 (1,) 导致 diff_bias batch=1，需要 expand 到 B
            target_B = h_variant.shape[0]
            target_K = h_variant.shape[2]
            target_L = h_variant.shape[3]
            diff_bias = diff_bias.expand(target_B, -1, target_K, target_L)  # (B, C, K, L)
            # 确保 h_stable 与 h_variant 的空间维度一致
            if h_stable.shape[2] != target_K or h_stable.shape[3] != target_L:
                h_stable = h_stable.reshape(target_B, h_stable.shape[1], target_K, target_L)
            h_concat = torch.cat([h_stable, h_variant, diff_bias], dim=1)  # (B, 3C, K, L)
            gate_stable = self.fusion_gate(h_concat)               # (B, C, K, L)
            h_fused = gate_stable * h_stable + (1.0 - gate_stable) * h_variant
        
        elif self.fusion_type == "attention":
            # 注意力融合
            q = self.fusion_query(h_stable)
            k = self.fusion_key(h_variant)
            v = self.fusion_value(h_variant)
            
            attn = torch.softmax(
                torch.sum(q * k, dim=1, keepdim=True) / math.sqrt(self.channels),
                dim=1
            )
            
            h_fused = h_stable + attn * v
        
        else:
            # 简单加和
            h_fused = h_stable + h_variant
        
        # 归一化
        h_fused = self.fusion_norm(h_fused)
        
        return h_fused

    def forward(self, x, cond_info, diffusion_step, strategy_type):
        """
        前向传播：频率增强的扩散去噪
        
        Args:
            x: (B, inputdim, K, L) - 输入（加噪数据）
            cond_info: (B, cond_dim, K, L) - 条件信息
            diffusion_step: (B,) - 扩散时间步
            strategy_type: (B,) - 策略类型
        Returns:
            out: (B, K, L) - 预测的噪声
        """
        B, inputdim, K, L = x.shape

        # 扩散时间步和策略嵌入（提前计算，供双编码器使用）
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        strategy_emb = self.strategy_embedding(strategy_type)

        # ============ 🔥 频率增强分支 ============
        if self.use_freq_enhancement:
            # Step 1: 提取时域信号
            if inputdim == 1:
                x_time = x.squeeze(1)  # (B, K, L)
            else:
                # 条件扩散：取noisy target
                x_time = x[:, 1, :, :]  # (B, K, L)
            
            # Step 2: 频域分解
            x_stable, x_variant = self.frequency_decomposition(x_time)
            
            # Step 3: 双编码器
            # 稳定分支：复用主干 residual_layers，skip 加权求和结果作为稳定特征
            # 变化分支：ModernTCN，分支权重由 fusion_gate 自适应决定
            h_stable = self.encode_stable_component(x_stable, cond_info, diffusion_emb, strategy_emb)
            h_variant = self.encode_variant_component(x_variant, diffusion_emb, cond_info)
            
            # Step 4: 双分支融合
            h_fused = self.fuse_dual_branch(h_stable, h_variant, diffusion_emb)
            
            # Step 5: 输出投影（直接从融合结果输出，不再重复走 residual_layers）
            agg = self.freq_to_output(h_fused)  # (B, channels, K, L)
            agg_flat = agg.reshape(B, self.channels, K * L)
            if torch.isnan(agg_flat).any() or torch.isinf(agg_flat).any():
                agg_flat = torch.clamp(agg_flat, -5.0, 5.0)
            out = self.output_projection1(agg_flat)
            out = F.relu(out)
            out = self.output_projection2(out)
            out = out.reshape(B, K, L)
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
            return out
        
        # ============ ⚪ 标准分支 ============
        else:
            x_flat = x.reshape(B, inputdim, K * L)
            xp = self.input_projection(x_flat)  # (B, channels, K*L)
            xp = F.relu(xp).reshape(B, self.channels, K, L)

        # ============ ⚪ 标准分支：残差块 + 输出投影 ============
        # 频率增强分支已在上方提前 return，此处仅服务于标准分支
        skip_sum = None
        cur = xp
        weights = torch.softmax(self.skip_weights.to(xp.device), dim=0)

        for i, layer in enumerate(self.residual_layers):
            cur, skip_conn = layer(cur, cond_info, diffusion_emb, strategy_emb)
            skip_conn = torch.clamp(skip_conn, -10.0, 10.0)
            w = weights[i]
            if skip_sum is None:
                skip_sum = skip_conn * w
            else:
                skip_sum = skip_sum + skip_conn * w
            del skip_conn

        agg = skip_sum if skip_sum is not None else torch.zeros_like(cur)

        agg_flat = agg.reshape(B, self.channels, K * L)
        if torch.isnan(agg_flat).any() or torch.isinf(agg_flat).any():
            agg_flat = torch.clamp(agg_flat, -5.0, 5.0)

        out = self.output_projection1(agg_flat)
        out = F.relu(out)
        out = self.output_projection2(out)
        out = out.reshape(B, K, L)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


