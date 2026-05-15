import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
from tqdm import tqdm
import random

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device,ratio = 0.7):
        super().__init__()
        self.device = device
        self.ratio = ratio
        self.target_dim = target_dim

        self.ddim_eta = 1
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        print("unconditional is")
        print(self.is_unconditional)
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        
        # 新增：时间戳嵌入维度配置
        self.timestamp_emb_dim = config["model"].get("timestamp_emb_dim", 32)
        self.emb_total_dim += self.timestamp_emb_dim  # 添加时间戳嵌入维度
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        
        # 新增：时间戳嵌入层
        self.timestamp_embed_layer = nn.Linear(5, self.timestamp_emb_dim)  # 5个时间特征：minute, hour, weekday, day, month
        


        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        # cumprod函数表示将之前的alpha连乘。这里的self.alpha实际上就是\overline \alpha
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        
        # ============ 🔥 双尺度时间步参数 (来自 DiffusionAD) ============
        self.condition_w = config_diff.get("condition_w", 1.0)  # 范数引导权重
        self.use_dual_scale = config_diff.get("use_dual_scale", False)  # 是否启用双尺度
        self.dual_scale_strategy = config_diff.get("dual_scale_strategy", "snr_adaptive")  # 采样策略
        
        # 预计算常用系数（用于范数引导）
        self.sqrt_alphas_cumprod = np.sqrt(self.alpha)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alpha)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alpha)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alpha - 1)
        
        # 基于信噪比的自适应采样参数
        self.low_snr_percentile = config_diff.get("low_snr_percentile", 0.2)
        self.high_snr_percentile = config_diff.get("high_snr_percentile", 0.8)
        
        # 预计算 SNR 值（SNR = alpha / (1 - alpha)）
        self.snr_values = self.alpha / (1.0 - self.alpha + 1e-8)
        
        # 🔥 在对数空间中做 percentile
        # 原因：log(SNR) 才是扩散模型的"真实时间刻度"
        # - 原始 SNR 分布极度偏态（最大值是中位数的 1000+ 倍）
        # - log(SNR) 分布近似均匀，符合信息损失的指数特性
        log_snr = np.log(self.snr_values + 1e-8)
        
        # 在对数空间中找阈值
        # 注意：低噪声对应高 SNR（高 log SNR），高噪声对应低 SNR（低 log SNR）
        log_snr_low_threshold = np.percentile(log_snr, self.high_snr_percentile * 100)   # 80% → 低噪声
        log_snr_high_threshold = np.percentile(log_snr, self.low_snr_percentile * 100)   # 20% → 高噪声
        
        # 找到 log(SNR) 最接近阈值的时间步
        self.low_snr_t = int(np.argmin(np.abs(log_snr - log_snr_low_threshold)))
        self.high_snr_t = int(np.argmin(np.abs(log_snr - log_snr_high_threshold)))
        
        if self.use_dual_scale:
            print(f"✅ 双尺度范数引导已启用:")
            if self.dual_scale_strategy == "snr_adaptive":
                log_snr = np.log(self.snr_values + 1e-8)
                print(f"   - 采样策略: 基于 log(SNR) 的自适应采样")
                print(f"   - 低噪声区间: t ∈ [0, {self.low_snr_t}]")
                print(f"     └─ SNR 范围: [{self.snr_values[0]:.2f}, {self.snr_values[self.low_snr_t]:.2f}]")
                print(f"     └─ log(SNR) 范围: [{log_snr[0]:.4f}, {log_snr[self.low_snr_t]:.4f}]")
                print(f"   - 高噪声区间: t ∈ [{self.high_snr_t}, {self.num_steps}]")
                print(f"     └─ SNR 范围: [{self.snr_values[self.high_snr_t]:.2f}, {self.snr_values[-1]:.6f}]")
                print(f"     └─ log(SNR) 范围: [{log_snr[self.high_snr_t]:.4f}, {log_snr[-1]:.4f}]")
                print(f"   - SNR 比例: {self.snr_values[self.low_snr_t] / (self.snr_values[self.high_snr_t] + 1e-8):.2f}x")
                print(f"   - 时间步分布: 低噪声 {self.low_snr_t} 步, 高噪声 {self.num_steps - self.high_snr_t} 步 (均衡 ✅)")
            else:
                print(f"   - 采样策略: 固定间距采样")
            print(f"   - normal_t 范围: [0, {self.less_t_range})")
            print(f"   - noisier_t 范围: [{self.less_t_range}, {self.noisier_t_range})")
                print(f"   - 固定间距 (gap): {self.dual_scale_gap}")
            print(f"   - 范数引导权重: {self.condition_w}")
        else:
            print(f"⚪ 使用标准 CSDI 扩散模型")

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    # 新增：时间戳嵌入方法
    def timestamp_embedding(self, timestamp_features):
        """
        将时间戳特征转换为嵌入向量
        timestamp_features: (B, L, 5) 或 (L, 5) - minute, hour, weekday, day, month
        """
        # 确保输入是3D格式 (B, L, 5)
        if timestamp_features.dim() == 2:
            # 如果是 (L, 5)，添加batch维度
            timestamp_features = timestamp_features.unsqueeze(0)
        elif timestamp_features.dim() != 3:
            raise ValueError(f"timestamp_features should have 2 or 3 dimensions, got {timestamp_features.dim()}")
        
        # 确保数据类型是float
        timestamp_features = timestamp_features.float()
        
        # 通过线性层转换
        timestamp_emb = self.timestamp_embed_layer(timestamp_features)  # (B, L, timestamp_emb_dim)
        timestamp_emb = torch.relu(timestamp_emb)  # 添加非线性激活
        
        return timestamp_emb

    def get_randmask(self, observed_mask,ratio = 0.7):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) #(b, *)
        for i in range(len(observed_mask)):
            # sample_ratio = np.random.rand()  # missing ratio
            sample_ratio = ratio  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            # 选择num_masked个数字，让它等于-1
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask,ratio=self.ratio)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask, timestamp_features=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # 新增：处理时间戳特征
        if timestamp_features is not None:
            
            # 处理时间戳特征：从 (100, 5) 转换为 (B, L, 5)
            if timestamp_features.dim() == 2:
                # 如果是 (L, 5)，需要扩展为 (B, L, 5)
                timestamp_features = timestamp_features.unsqueeze(0).expand(B, -1, -1)
            
            timestamp_embed = self.timestamp_embedding(timestamp_features)  # (B, L, timestamp_emb_dim)
            # 扩展维度以匹配其他嵌入
            timestamp_embed = timestamp_embed.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, timestamp_emb_dim)
            
            # 智能融合策略
            # 1. 首先拼接所有嵌入
            concat_embeddings = torch.cat([time_embed, feature_embed, timestamp_embed], dim=-1)  # (B,L,K,emb_total_dim)
            
            # 2. 重塑为序列格式用于注意力机制
            seq_len = L * K
            embeddings_seq = concat_embeddings.view(B, seq_len, -1)  # (B, L*K, emb_total_dim)
            
            # 3. 使用局部+稀疏注意力机制进行信息交互
            # 计算注意力掩码（局部窗口）
            window_size = 32  # 局部注意力窗口大小
            device = embeddings_seq.device
            seq_len = embeddings_seq.size(1)
            
            # 创建局部注意力掩码
            local_mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                local_mask[i, start:end] = 0
            
            # 计算注意力分数
            q = embeddings_seq
            k = embeddings_seq
            v = embeddings_seq
            
            # 计算注意力权重
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float))
            attn_weights = attn_weights + local_mask.unsqueeze(0)  # 添加局部掩码
            
            # 稀疏化：只保留top-k个注意力权重
            sparsity_ratio = 0.1  # 保留10%的注意力权重
            top_k = int(seq_len * sparsity_ratio)
            
            # 对每个查询位置，只保留top-k个最大的注意力权重
            top_values, _ = torch.topk(attn_weights, k=top_k, dim=-1)
            threshold = top_values[..., -1, None]
            sparse_mask = (attn_weights < threshold)
            attn_weights = attn_weights.masked_fill(sparse_mask, float('-inf'))
            
            # 应用softmax
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # 计算输出
            attended_embeddings = torch.matmul(attn_weights, v)
            
            # 4. 重塑回原始维度
            side_info = attended_embeddings.view(B, L, K, -1)  
            
        else:
            # 如果没有时间戳特征，使用原来的方式
            side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type,
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type=strategy_type, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def sample_dual_timesteps_snr_adaptive(self, batch_size, is_train=1):
        """
        基于信噪比的自适应采样
        从低噪声和高噪声区间采样，使得两个尺度的SNR比例接近目标值
        
        Returns:
            normal_t: 低噪声时间步 (B,)
            noisier_t: 高噪声时间步 (B,)
        """
        if is_train == 1:
            # 训练时：从两个区间随机采样
            # 从低噪声区间采样
            normal_t = torch.randint(0, self.low_snr_t + 1, (batch_size,), device=self.device)
            
            # 从高噪声区间采样
            noisier_t = torch.randint(self.high_snr_t, self.num_steps, (batch_size,), device=self.device)
        else:
            # 验证时：使用固定时间步
            normal_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * (self.low_snr_t // 2)
            noisier_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * ((self.high_snr_t + self.num_steps) // 2)
        
        return normal_t, noisier_t

    def extract(self, arr, timesteps, broadcast_shape):
        """提取特定时间步的系数（用于双尺度方法）"""
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def predict_x_0_from_eps(self, x_t, t, eps):
        """从噪声预测 x_0（用于双尺度方法）"""
        return (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def sample_q(self, x_0, t, noise):
        """前向扩散: q(x_t | x_0)（用于双尺度方法）"""
        return (self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
                self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)

    # ============ 🔥 核心创新：双尺度范数引导单步去噪 ============
    def norm_guided_one_step_denoising(self, observed_data, cond_mask, observed_mask, 
                                       side_info, is_train, strategy_type):
        """
        双尺度时间步采样 + 范数引导单步去噪
        参考 DiffusionAD 的 norm_guided_one_step_denoising 方法
        
        🔥 核心逻辑完全按照论文实现，只是采样方式改为 SNR 自适应
        
        流程:
        1. 采样两个不同尺度的时间步 (normal_t 和 noisier_t)
        2. 在两个尺度上分别计算损失
        3. 从 noisier 尺度预测 x_0
        4. 将 noisier 尺度的 x_0 投影到 normal 尺度
        5. 计算范数引导的噪声估计
        6. 从引导后的噪声预测最终的 x_0
        
        Returns:
            total_loss: 联合损失
            pred_x_0_norm_guided: 范数引导后的重建结果
            normal_t: normal 尺度的时间步
            noisier_t: noisier 尺度的时间步
        """
        B, K, L = observed_data.shape
        
        # 1. 双尺度时间步采样
        # 🔥 改进：使用 SNR 自适应采样而非固定间距
        if self.dual_scale_strategy == "snr_adaptive":
            normal_t, noisier_t = self.sample_dual_timesteps_snr_adaptive(B, is_train)
        else:
            # 回退到固定间距策略
        if is_train == 1:
            normal_t = torch.randint(0, self.less_t_range, (B,), device=self.device)
                noisier_t = torch.randint(self.less_t_range, self.num_steps, (B,), device=self.device)
        else:
            normal_t = torch.ones(B, dtype=torch.long, device=self.device) * (self.less_t_range // 2)
                noisier_t = torch.ones(B, dtype=torch.long, device=self.device) * ((self.less_t_range + self.num_steps) // 2)
        
        # 2. 在两个尺度上分别计算损失
        # 这里直接调用 calc_loss 方法，和论文的 self.calc_loss 逻辑一致
        target_mask = observed_mask - cond_mask
        
        # Normal scale
        noise_normal = torch.randn_like(observed_data)
        current_alpha_normal = self.alpha_torch[normal_t]
        x_normal_t = (current_alpha_normal ** 0.5) * observed_data + (1.0 - current_alpha_normal) ** 0.5 * noise_normal
        total_input_normal = self.set_input_to_diffmodel(x_normal_t, observed_data, cond_mask)
        estimate_noise_normal = self.diffmodel(total_input_normal, side_info, normal_t, strategy_type)
        
        residual_normal = (noise_normal - estimate_noise_normal) * target_mask
        normal_loss = (residual_normal ** 2).sum() / (target_mask.sum() if target_mask.sum() > 0 else 1)
        
        # Noisier scale
        noise_noisier = torch.randn_like(observed_data)
        current_alpha_noisier = self.alpha_torch[noisier_t]
        x_noisier_t = (current_alpha_noisier ** 0.5) * observed_data + (1.0 - current_alpha_noisier) ** 0.5 * noise_noisier
        total_input_noisier = self.set_input_to_diffmodel(x_noisier_t, observed_data, cond_mask)
        estimate_noise_noisier = self.diffmodel(total_input_noisier, side_info, noisier_t, strategy_type)
        
        residual_noisier = (noise_noisier - estimate_noise_noisier) * target_mask
        noisier_loss = (residual_noisier ** 2).sum() / (target_mask.sum() if target_mask.sum() > 0 else 1)
        
        # 3. 从 noisier 尺度预测 x_0
        pred_x_0_noisier = self.predict_x_0_from_eps(x_noisier_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        
        # 4. 将 noisier 尺度的 x_0 投影到 normal 尺度
        # 🔥 关键：使用 noise_normal 来投影，保持一致性
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, noise_normal)
        
        # 5. 计算总损失
        loss = normal_loss + noisier_loss
        
        # 6. 范数引导：计算引导后的噪声估计
        sqrt_one_minus_alpha_normal = self.extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_normal_t.shape)
        estimate_noise_hat = estimate_noise_normal - sqrt_one_minus_alpha_normal * self.condition_w * (pred_x_t_noisier - x_normal_t)
        
        # 7. 从引导后的噪声预测最终的 x_0
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)
        
        return loss, pred_x_0_norm_guided, normal_t, noisier_t

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type, set_t=-1
    ):
        """
        计算损失函数
        根据 self.use_dual_scale 自动选择标准方法或双尺度方法
        """
        # 如果启用双尺度，使用范数引导方法
        if self.use_dual_scale:
            loss, _, _, _ = self.norm_guided_one_step_denoising(
                observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type
            )
            return loss
        
        # 否则使用标准 CSDI 方法
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
       
        predicted = self.diffmodel(total_input, side_info, t, strategy_type)  # (B,K,L)

        # 此处的condition mask全部为0
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, strategy_type):
        """
        推理采样方法
        根据 self.use_dual_scale 自动选择标准 DDPM 或双尺度范数引导采样
        """
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # 生成初始噪声历史（用于无条件模型）
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            # 🔥 双尺度范数引导采样
            if self.use_dual_scale:
                for t in range(self.num_steps - 1, -1, -1):
                    # 构造输入
                    if self.is_unconditional == True:
                        diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                        diff_input = diff_input.unsqueeze(1)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                    
                    # 预测当前时间步的噪声
                    t_tensor = torch.tensor([t]).to(self.device)
                    predicted = self.diffmodel(diff_input, side_info, t_tensor, strategy_type)
                    
                    # 🔥 范数引导机制（仅在高噪声区间应用）
                    if self.dual_scale_strategy == "snr_adaptive":
                        if t >= self.high_snr_t:
                            # 基于SNR的映射找到对应的 normal_t
                            current_snr = self.snr_values[t]
                            target_snr_ratio = 0.5  # 目标SNR是当前SNR的50%
                            target_snr = current_snr * target_snr_ratio
                            snr_diff = np.abs(self.snr_values[:self.low_snr_t] - target_snr)
                            normal_t = int(np.argmin(snr_diff))
                        
                        normal_t_tensor = torch.tensor([normal_t]).to(self.device)
                        
                        # 从当前预测重建 x_0
                        current_alpha = self.alpha_torch[t_tensor]
                        pred_x_0 = (current_sample - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha ** 0.5)
                        pred_x_0 = pred_x_0.clamp(-1, 1)
                        
                        # 将 x_0 投影到 normal_t
                        # 🔥 修复：用模型预测的 predicted 作为投影噪声，与论文一致
                        # 论文: pred_x_t_noisier = sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)
                        normal_alpha = self.alpha_torch[normal_t_tensor]
                        pred_x_t_normal = (normal_alpha ** 0.5) * pred_x_0 + (1.0 - normal_alpha) ** 0.5 * predicted
                        
                        # 🔥 修复：引导系数使用 normal_t 对应的 sqrt_one_minus_alphas_cumprod，与训练一致
                        # 论文: extract(sqrt_one_minus_alphas_cumprod, normal_t, ...)
                        sqrt_one_minus_alpha_normal = float(self.sqrt_one_minus_alphas_cumprod[normal_t])
                        predicted = predicted - sqrt_one_minus_alpha_normal * self.condition_w * (pred_x_t_normal - current_sample)
                    else:
                        # 固定间距策略
                        if t >= self.less_t_range:
                            normal_t = max(0, t - self.dual_scale_gap)
                            normal_t_tensor = torch.tensor([normal_t]).to(self.device)
                            
                            current_alpha = self.alpha_torch[t_tensor]
                            pred_x_0 = (current_sample - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha ** 0.5)
                            pred_x_0 = pred_x_0.clamp(-1, 1)
                            
                            # 🔥 修复：用模型预测的 predicted 作为投影噪声
                            normal_alpha = self.alpha_torch[normal_t_tensor]
                            pred_x_t_normal = (normal_alpha ** 0.5) * pred_x_0 + (1.0 - normal_alpha) ** 0.5 * predicted
                        
                            # 🔥 修复：引导系数使用 normal_t 对应的 sqrt_one_minus_alphas_cumprod
                            sqrt_one_minus_alpha_normal = float(self.sqrt_one_minus_alphas_cumprod[normal_t])
                            predicted = predicted - sqrt_one_minus_alpha_normal * self.condition_w * (pred_x_t_normal - current_sample)
                    
                    # 标准 DDPM 去噪步骤
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                        current_sample += sigma * noise
            
            # ⚪ 标准 DDPM 采样
            else:
                for t in range(self.num_steps - 1, -1, -1):
                    if self.is_unconditional == True:
                        diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                        diff_input = diff_input.unsqueeze(1)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                    
                    predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), strategy_type)
                    
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                        current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        
        return imputed_samples


    def ddim_impute(self, observed_data, cond_mask, side_info, n_samples,ddim_eta=1,ddim_steps=10):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)


        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    # 这里的noisy_cond_history就是对整个数据片上的所有数据进行了加噪
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            ddim_timesteps = ddim_steps
            c = self.num_steps // ddim_timesteps
            ddim_timesteps_sequence = np.asarray(list(range(0, self.num_steps, c)))
            ddim_timesteps_previous_sequence = np.append(
                np.array([0]) , ddim_timesteps_sequence[: -1]
            )

            for step_number in range(ddim_timesteps - 1 , -1, -1):
                t = ddim_timesteps_sequence[step_number]
                previous_t =  ddim_timesteps_previous_sequence[step_number]

                at = torch.tensor(self.alpha[t]).to(self.device)
                at_next = torch.tensor(self.alpha[previous_t]).to(self.device)

                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                xt = diff_input
                et = self.diffmodel(xt, side_info, torch.tensor([t]).to(self.device))
                x0_t = (current_sample - et * (1 - at).sqrt()) / at.sqrt()

                c1 = (
                        ddim_eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                current_sample = at_next.sqrt() * x0_t + c1 * torch.randn_like(current_sample) + c2 * et

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


    def get_middle_impute_value(self, observed_data, cond_mask, side_info, n_samples, strategy_type):
        """
        获取中间步骤的重建值（用于窗口技巧评估）
        根据 self.use_dual_scale 自动选择标准 DDPM 或双尺度范数引导采样
        """
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        imputed_middle_samples = torch.zeros(B, self.num_steps, K, L).to(self.device)

        for i in range(n_samples):
            # 生成初始噪声历史（用于无条件模型）
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            # 🔥 双尺度范数引导采样
            if self.use_dual_scale:
                for t in range(self.num_steps - 1, -1, -1):
                    # 构造输入
                    if self.is_unconditional == True:
                        diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                        diff_input = diff_input.unsqueeze(1)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                    
                    # 预测当前时间步的噪声
                    t_tensor = torch.tensor([t]).to(self.device)
                    predicted = self.diffmodel(diff_input, side_info, t_tensor, strategy_type)
                    
                    # 🔥 范数引导机制（仅在高噪声区间应用）
                    if self.dual_scale_strategy == "snr_adaptive":
                        if t >= self.high_snr_t:
                            # 🔥 改进：基于SNR的映射而非线性映射
                            current_snr = self.snr_values[t]
                            target_snr_ratio = 0.5
                            target_snr = current_snr * target_snr_ratio
                            snr_diff = np.abs(self.snr_values[:self.low_snr_t] - target_snr)
                            normal_t = int(np.argmin(snr_diff))
                        
                        normal_t_tensor = torch.tensor([normal_t]).to(self.device)
                        
                        # 从当前预测重建 x_0
                        current_alpha = self.alpha_torch[t_tensor]
                        pred_x_0 = (current_sample - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha ** 0.5)
                        pred_x_0 = pred_x_0.clamp(-1, 1)
                        
                        # 将 x_0 投影到 normal_t
                        # 🔥 修复：用模型预测的 predicted 作为投影噪声，与论文一致
                        # 论文: pred_x_t_noisier = sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)
                        normal_alpha = self.alpha_torch[normal_t_tensor]
                        pred_x_t_normal = (normal_alpha ** 0.5) * pred_x_0 + (1.0 - normal_alpha) ** 0.5 * predicted
                        
                        # 🔥 修复：引导系数使用 normal_t 对应的 sqrt_one_minus_alphas_cumprod，与训练一致
                        sqrt_one_minus_alpha_normal = float(self.sqrt_one_minus_alphas_cumprod[normal_t])
                        predicted = predicted - sqrt_one_minus_alpha_normal * self.condition_w * (pred_x_t_normal - current_sample)
                    else:
                        # 固定间距策略
                        if t >= self.less_t_range:
                            normal_t = max(0, t - self.dual_scale_gap)
                            normal_t_tensor = torch.tensor([normal_t]).to(self.device)
                            
                            current_alpha = self.alpha_torch[t_tensor]
                            pred_x_0 = (current_sample - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha ** 0.5)
                            pred_x_0 = pred_x_0.clamp(-1, 1)
                            
                            # 🔥 修复：用模型预测的 predicted 作为投影噪声
                            normal_alpha = self.alpha_torch[normal_t_tensor]
                            pred_x_t_normal = (normal_alpha ** 0.5) * pred_x_0 + (1.0 - normal_alpha) ** 0.5 * predicted
                        
                            # 🔥 修复：引导系数使用 normal_t 对应的 sqrt_one_minus_alphas_cumprod
                            sqrt_one_minus_alpha_normal = float(self.sqrt_one_minus_alphas_cumprod[normal_t])
                            predicted = predicted - sqrt_one_minus_alpha_normal * self.condition_w * (pred_x_t_normal - current_sample)
                    
                    # 标准 DDPM 去噪步骤
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                        current_sample += sigma * noise
                    
                    # 保存中间步骤结果
                    imputed_middle_samples[:, t] = current_sample.detach()
            
            # ⚪ 标准 DDPM 采样
            else:
                for t in range(self.num_steps - 1, -1, -1):
                    if self.is_unconditional == True:
                        diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                        diff_input = diff_input.unsqueeze(1)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                    
                    predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), strategy_type)
                    
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                        current_sample += sigma * noise
                    
                    # 保存中间步骤结果
                    imputed_middle_samples[:, t] = current_sample.detach()

            imputed_samples[:, i] = current_sample.detach()
        
        return imputed_samples, imputed_middle_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            strategy_type,
            timestamp_features
        ) = self.process_data(batch)

         # 🔥 不使用掩码策略，直接在完整数据上训练双尺度扩散模型
        # 将 cond_mask 设置为全零，表示没有条件信息（无条件扩散）
        cond_mask = torch.zeros_like(observed_mask)
        
        #self.target_strategy = "random"
        #if is_train == 0:
        #    cond_mask = gt_mask
        #elif self.target_strategy != "random":
        #    cond_mask = self.get_hist_mask(
        #        observed_mask, for_pattern_mask=for_pattern_mask
        #    )
        #else:
        #    cond_mask = self.get_randmask(observed_mask,ratio=self.ratio)
            #
            # cond_mask = torch.zeros_like(observed_mask)
            # cond_mask = self.get_random_mask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, timestamp_features)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        
        # 计算双尺度扩散损失（在完整数据上）
        diffusion_loss = loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, strategy_type = strategy_type)
        
        return diffusion_loss

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            strategy_type,
            timestamp_features
           
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, timestamp_features)

            print(f"strategy type in evaluate is {strategy_type}")
            samples = self.impute(observed_data, cond_mask, side_info, n_samples, strategy_type)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        # 此处target_mask给的是那些待预测的点
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def get_middle_evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            strategy_type,
            timestamp_features
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, timestamp_features)

            samples,imputed_middle_samples = self.get_middle_impute_value(observed_data, cond_mask, side_info, n_samples, strategy_type)

            print("shape of imputed middle samples is")
            print(imputed_middle_samples.shape)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp, imputed_middle_samples

    def ddim_evaluate(self, batch, n_samples,ddim_eta=1,ddim_steps=10):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            strategy_type,
            timestamp_features
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, timestamp_features)

            samples = self.ddim_impute(observed_data, cond_mask, side_info, n_samples,ddim_eta=ddim_eta,ddim_steps=ddim_steps)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        timestamp_features = batch["timestamp_features"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            timestamp_features
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35,ratio = 0.7):
        super(CSDI_Physio, self).__init__(target_dim, config, device,ratio)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        timestamp_features = batch["timestamp_features"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        strategy_type = batch['strategy_type'].to(self.device).long()

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            strategy_type,
            timestamp_features
        )
