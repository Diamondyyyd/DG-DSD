import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
import time 
from torch.utils.data import Dataset, DataLoader
from ddecomposition.decomposition import DynamicDecomposition
from ddecomposition.preprocess import getData
from earlystop import EarlyStop  # 从D3R中导入早停机制

class DecompositionDataset(Dataset):
    def __init__(self, data, time, stable, label, window_size):
        """与D3R保持一致的数据集类
        
        参数:
            data: 原始数据
            time: 时间特征
            stable: 稳定分量
            label: 标签
            window_size: 窗口大小
        """
        self.data = data
        self.time = time
        self.stable = stable
        self.label = label
        self.window_size = window_size

    def __getitem__(self, index):
        data = self.data[index: index + self.window_size, :]
        time = self.time[index: index + self.window_size, :]
        stable = self.stable[index: index + self.window_size, :]
        
      
        if len(self.label.shape) == 1:
           
            label = self.label[index: index + self.window_size]
        else:
            
            label = self.label[index: index + self.window_size, :]
        
        return data, time, stable, label

    def __len__(self):
        return len(self.data) - self.window_size + 1

class D3RDecomposer:
    def __init__(self, device="cuda:0", window_size=100, feature_dim=38, time_dim=5):
        """
        初始化D3R分解器（纯时域分解）
        
        Args:
            device: 设备
            window_size: 窗口大小
            feature_dim: 特征维度
            time_dim: 时间特征维度
        """
        self.device = device
        
        # 初始化动态分解模型（纯时域）
        self.model = DynamicDecomposition(
            window_size=window_size,
            model_dim=512,
            ff_dim=2048,
            atten_dim=64,
            feature_num=feature_dim,
            time_num=time_dim,
            block_num=2,
            head_num=8,
            dropout=0.6,
            d=30
        ).to(device)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    def train(self, train_loader, valid_loader=None, epochs=100, result_folder="decomposer_results", patience=10):
        """
        训练分解模型（纯时域）
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            epochs: 训练轮数
            result_folder: 结果保存文件夹
            patience: 早停耐心值
        """
        os.makedirs(result_folder, exist_ok=True)
        
        # 初始化早停
        early_stopping = EarlyStop(path=os.path.join(result_folder, 'best_decomposer.pth'), patience=patience)
        
        for epoch in range(epochs):
            start = time.time()
            
            # 训练阶段
            self.model.train()
            train_losses = []
            
            for batch_data, batch_time, batch_stable, _ in tqdm(train_loader):
                batch_data = batch_data.float().to(self.device)
                batch_time = batch_time.float().to(self.device)
                batch_stable = batch_stable.float().to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播：返回稳定成分和趋势成分
                stable, trend = self.model(batch_data, batch_time)
                
                # 损失一：stable 监督损失（有真实标签）
                loss_stable = self.criterion(stable, batch_stable)
                
                # 损失二：trend 一阶差分平滑正则
                # 鼓励趋势在相邻时间步之间平滑变化，为 offset_bias 和 log_tau 提供梯度驱动
                # trend: [B, L, F]，对时间维度(dim=1)求一阶差分
                trend_diff = trend[:, 1:, :] - trend[:, :-1, :]  # [B, L-1, F]
                loss_trend = torch.mean(trend_diff ** 2)
                
                # 总损失：stable 监督为主，trend 平滑正则为辅
                # 权重比例可根据实验调整，建议 loss_trend 权重范围 0.05~0.2
                loss = 0.9 * loss_stable + 0.1 * loss_trend
                
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # 验证阶段
            if valid_loader is not None:
                self.model.eval()
                valid_losses = []
                
                with torch.no_grad():
                    for batch_data, batch_time, batch_stable, _ in tqdm(valid_loader):
                        batch_data = batch_data.float().to(self.device)
                        batch_time = batch_time.float().to(self.device)
                        batch_stable = batch_stable.float().to(self.device)
                        
                        # 前向传播
                        stable, trend = self.model(batch_data, batch_time)
                        
                        # 计算损失
                        loss = self.criterion(stable, batch_stable)
                        valid_losses.append(loss.item())
                
                avg_valid_loss = np.average(valid_losses)
                
                # 早停检查
                early_stopping(avg_valid_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
            
            avg_train_loss = np.average(train_losses)
            end = time.time()
            
            # 显示训练进度
            valid_loss_str = ""
            if valid_loader is not None:
                valid_loss_str = f" Valid Loss: {avg_valid_loss:.6f}"
            
            print(f'Epoch: {epoch + 1} [Time-domain] || Train Loss: {avg_train_loss:.6f}{valid_loss_str} || Cost: {end - start:.4f}s')
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(os.path.join(result_folder, 'best_decomposer.pth')))
        
    def evaluate(self, test_loader):
        """
        评估模型（纯时域）
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            avg_test_loss: 平均测试损失
        """
        self.model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch_data, batch_time, batch_stable, _ in tqdm(test_loader, desc="Evaluating"):
                batch_data = batch_data.float().to(self.device)
                batch_time = batch_time.float().to(self.device)
                batch_stable = batch_stable.float().to(self.device)
                
                # 前向传播
                stable, trend = self.model(batch_data, batch_time)
                
                # 计算损失
                loss = self.criterion(stable, batch_stable)
                test_losses.append(loss.item())
        
        avg_test_loss = np.average(test_losses)
        print(f"Test Loss [Time-domain]: {avg_test_loss:.6f}")
        return avg_test_loss

def main():
    parser = argparse.ArgumentParser(description="D3R Decomposer Training")
    parser.add_argument("--dataset", type=str, default="SMD", help="数据集名称")
    parser.add_argument("--device", type=int, default=0, help="设备编号")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=12, help="批大小")
    parser.add_argument("--window_length", type=int, default=100, help="窗口长度")
    parser.add_argument("--result_folder", type=str, default="decomposer_results", help="结果保存文件夹")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    
    args = parser.parse_args()
    
    # 设置设备
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*80}")
    print(f"模式: 纯时域分解")
    print(f"{'='*80}\n")
    
    # 获取预处理数据
    data = getData(
        path='./data/Machine/',
        dataset=args.dataset,
        period=1440,
        train_rate=0.8
    )
    
    # 创建数据集
    train_set = DecompositionDataset(
        data=data['train_data'],
        time=data['train_time'],
        stable=data['train_stable'],
        label=data['train_label'],
        window_size=args.window_length
    )
    
    valid_set = DecompositionDataset(
        data=data['valid_data'],
        time=data['valid_time'],
        stable=data['valid_stable'],
        label=data['valid_label'],
        window_size=args.window_length
    )
    
    test_set = DecompositionDataset(
        data=data['test_data'],
        time=data['test_time'],
        stable=data['test_stable'],
        label=data['test_label'],
        window_size=args.window_length
    )
    
    
    print(f"训练集有效样本数: {len(train_set)}")
    print(f"验证集有效样本数: {len(valid_set)}")
    print(f"测试集有效样本数: {len(test_set)}")
    print(f"批次大小: {args.batch_size}")
    print(f"预计训练批次数: {len(train_set) // args.batch_size + (1 if len(train_set) % args.batch_size > 0 else 0)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    
    print(f"训练数据加载器批次数: {len(train_loader)}")
    print(f"验证数据加载器批次数: {len(valid_loader)}")
    print(f"测试数据加载器批次数: {len(test_loader)}")
    
    # 创建结果文件夹
    result_folder = os.path.join(args.result_folder, args.dataset)
    os.makedirs(result_folder, exist_ok=True)
    
    
    # 创建分解器（纯时域）
    decomposer = D3RDecomposer(
        device=device,
        window_size=args.window_length,
        feature_dim=data['train_data'].shape[1],
        time_dim=data['train_time'].shape[1]
    )
    
    # 训练模型
    decomposer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        result_folder=result_folder,
        patience=args.patience
    )
    
    # 评估模型
    decomposer.evaluate(test_loader)
    
    print("训练和评估完成！")

if __name__ == "__main__":
    main()
