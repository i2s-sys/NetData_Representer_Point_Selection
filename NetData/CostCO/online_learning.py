"""
Costco 在线学习预测 - 主文件

基于 Costco 模型的在线学习，实现增量训练和预测

方案：
1. 初始训练：使用前2000列作为历史数据
2. 预测第2001列
3. 增量更新：加入第2001列数据继续训练
4. 预测第2002列
5. 重复步骤3-4，直到预测完所有数据

使用方法:
    cd CostCO
    python online_learning.py --dataset Geant_23_23_3000

流程:
1. 转换数据为矩阵 (在 handleData/ 目录)
2. 运行在线学习预测 (在 CostCO/ 目录)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys


# 添加 handleData 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'handleData'))


class CostcoModel(nn.Module):
    """
    Costco 时间序列预测模型
    
    使用 LSTM 编码器 + 全连接层
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(CostcoModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (batch_size, seq_len, input_size)
        
        返回:
            predictions: (batch_size, input_size, 1)
        """
        # LSTM 编码
        lstm_out, _ = self.lstm(x)
        
        # 全连接输出（只取最后时间步）
        predictions = self.fc(lstm_out[:, -1, :])
        
        return predictions


class OnlinePredictor:
    """
    Costco 在线预测器
    
    实现滑动窗口和增量训练
    """
    def __init__(self, matrix_path, initial_window=2000, window_size=100,
                 hidden_size=128, lr=1e-3, epochs=5, device=None):
        """
        参数:
            matrix_path: 矩阵文件路径 (num_links, num_time)
            initial_window: 初始训练窗口大小（前N列）
            window_size: 滑动窗口大小
            hidden_size: LSTM 隐藏层大小
            lr: 学习率
            epochs: 每次更新的训练轮数
            device: 计算设备
        """
        print(f"\n{'='*70}")
        print("初始化 Costco 在线预测器")
        print(f"{'='*70}")

        # 加载矩阵 (链路, 时间)
        print(f"\n加载矩阵: {matrix_path}")
        self.matrix = np.load(matrix_path)
        self.num_links, self.num_time = self.matrix.shape

        print(f"  形状: {self.matrix.shape}")
        print(f"  链路数: {self.num_links}")
        print(f"  时间点数: {self.num_time}")

        # 归一化
        self.max_val = self.matrix.max()
        self.matrix_norm = self.matrix / self.max_val
        print(f"  数据范围: [{self.matrix.min():.2f}, {self.matrix.max():.2f}]")

        # 参数
        self.initial_window = initial_window
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.hidden_size = hidden_size

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"  设备: {self.device}")

        # 初始化模型
        self.model = CostcoModel(
            input_size=self.num_links,  # 链路数作为输入特征
            hidden_size=hidden_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        print(f"\n模型设置:")
        print(f"  隐藏层大小: {hidden_size}")
        print(f"  学习率: {lr}")
        print(f"  初始窗口: {initial_window} (列 0-{initial_window-1})")
        print(f"  滑动窗口: {window_size}")
        print(f"  训练轮数: {epochs} (每次更新)")

    def prepare_data(self, end_col):
        """
        准备训练数据
        
        参数:
            end_col: 结束列索引（不包含）
        
        返回:
            X: (seq_len, num_links) 时间序列
            y: (num_links,) 目标值
        """
        # 取最后 window_size 列作为训练数据
        start_col = max(0, end_col - self.window_size)
        X = self.matrix_norm[:, start_col:end_col].T  # (window_size, num_links)
        y = self.matrix_norm[:, end_col]  # (num_links,)
        
        return X, y

    def train(self, X, y, epochs=None):
        """
        训练模型
        
        参数:
            X: (seq_len, num_links)
            y: (num_links,)
            epochs: 训练轮数
        
        返回:
            loss: 损失值
        """
        if epochs is None:
            epochs = self.epochs

        self.model.train()

        # 转换为张量
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # (1, seq_len, num_links)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)  # (num_links, 1)

        for epoch in range(epochs):
            # 前向传播
            predictions = self.model(X_tensor)  # (1, num_links, 1)

            # 计算损失
            loss = self.criterion(predictions, y_tensor)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def predict(self, X):
        """
        预测
        
        参数:
            X: (seq_len, num_links)
        
        返回:
            predictions: (num_links,)
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().squeeze()  # (num_links,)

    def online_predict(self):
        """
        执行在线学习预测
        
        流程:
        1. 使用前 initial_window 列训练初始模型
        2. 预测第 initial_window + 1 列
        3. 加入第 initial_window + 1 列数据，继续训练
        4. 预测第 initial_window + 2 列
        5. 重复 3-4，直到预测完所有数据
        
        返回:
            predictions: (num_links, num_predictions) 预测结果
            true_values: (num_links, num_predictions) 真实值
        """
        print(f"\n{'='*70}")
        print("开始在线学习预测")
        print(f"{'='*70}")

        # ========== 阶段1: 初始训练 ==========
        print(f"\n【阶段1】初始训练")
        print(f"使用列 0-{self.initial_window-1} 作为历史数据")
        print(f"{'-'*70}")

        # 准备初始训练数据（使用整个初始窗口）
        X_init = self.matrix_norm[:, :self.initial_window].T  # (initial_window, num_links)
        y_init = self.matrix_norm[:, self.initial_window]  # (num_links,)

        print(f"训练数据: X形状={X_init.shape}, y形状={y_init.shape}")

        # 初始训练
        print(f"开始初始训练...")
        for epoch in range(self.epochs * 2):  # 初始训练多一些轮数
            loss = self.train(X_init, y_init, epochs=1)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs*2}, Loss: {loss:.6f}")

        # ========== 阶段2: 在线预测和增量更新 ==========
        print(f"\n【阶段2】在线预测 + 增量更新")
        print(f"预测列 {self.initial_window+1} 到 {self.num_time-1}")
        print(f"{'-'*70}")

        # 存储结果
        predictions = []
        true_values = []
        losses = []

        # 预测起始列
        predict_start = self.initial_window + 1

        for pred_col in range(predict_start, self.num_time):
            # ----- 预测 -----
            # 准备预测数据（最近 window_size 列）
            X_pred, _ = self.prepare_data(pred_col)
            pred = self.predict(X_pred)
            pred_denorm = pred * self.max_val

            # 获取真实值
            true_val = self.matrix[:, pred_col]

            # 计算损失
            loss = np.mean((pred_denorm - true_val) ** 2)

            # 存储结果
            predictions.append(pred_denorm)
            true_values.append(true_val)
            losses.append(loss)

            # 打印进度
            if (pred_col - predict_start + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"  预测列 {pred_col}/{self.num_time-1}, "
                      f"最近100列平均损失: {avg_loss:.4f}")

            # ----- 增量更新 -----
            # 准备更新训练数据（加入刚才预测的列）
            # 使用滑动窗口：保持 window_size 个最近的数据
            update_start = max(0, pred_col - self.window_size)
            X_update, y_update = self.prepare_data(pred_col + 1)  # 包含 pred_col 列

            # 增量训练（训练较少的 epoch）
            update_loss = self.train(X_update, y_update, epochs=self.epochs // 2)

        # 转换为 numpy 数组
        predictions = np.array(predictions).T  # (num_links, num_predictions)
        true_values = np.array(true_values).T  # (num_links, num_predictions)
        losses = np.array(losses)

        print(f"\n{'='*70}")
        print("在线预测完成！")
        print(f"预测列数: {predictions.shape[1]}")
        print(f"{'='*70}")

        # 评估结果
        self.evaluate_results(predictions, true_values, losses, predict_start)

        return predictions, true_values

    def evaluate_results(self, predictions, true_values, losses, predict_start):
        """
        评估预测结果
        """
        print(f"\n{'='*70}")
        print("预测结果评估")
        print(f"{'='*70}\n")

        # 计算指标
        mae = np.mean(np.abs(predictions - true_values))
        rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
        nmae = np.sum(np.abs(predictions - true_values)) / np.sum(np.abs(true_values))
        nrmse = np.sqrt(np.sum((predictions - true_values) ** 2) / np.sum(true_values ** 2))

        print(f"整体指标:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  NMAE: {nmae:.4f}")
        print(f"  NRMSE: {nrmse:.4f}")

        print(f"\n损失统计:")
        print(f"  平均损失: {np.mean(losses):.4f}")
        print(f"  最小损失: {np.min(losses):.4f}")
        print(f"  最大损失: {np.max(losses):.4f}")
        print(f"  中位数损失: {np.median(losses):.4f}")

        # 随时间变化的损失
        print(f"\n损失随时间变化:")
        for i in range(0, len(losses), len(losses) // 10):
            if i < len(losses):
                print(f"  列 {predict_start + i}: {losses[i]:.4f}")

    def save_results(self, predictions, true_values, output_dir='./output'):
        """
        保存预测结果
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存预测结果
        pred_path = os.path.join(output_dir, 'predictions.npy')
        np.save(pred_path, predictions)
        print(f"\n✓ 预测结果已保存: {pred_path}")

        # 保存真实值
        true_path = os.path.join(output_dir, 'true_values.npy')
        np.save(true_path, true_values)
        print(f"✓ 真实值已保存: {true_path}")

        return pred_path, true_path


def main():
    parser = argparse.ArgumentParser(description='Costco 在线学习预测')
    parser.add_argument('--dataset', type=str, default='Geant_23_23_3000',
                        help='数据集名称')
    parser.add_argument('--matrix_dir', type=str, default='./handleData/output',
                        help='矩阵目录 (默认: ./handleData/output)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='预测结果输出目录 (默认: ./output)')
    parser.add_argument('--initial_window', type=int, default=2000,
                        help='初始训练窗口大小 (默认: 2000)')
    parser.add_argument('--window_size', type=int, default=100,
                        help='滑动窗口大小 (默认: 100)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='LSTM隐藏层大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='每次更新的训练轮数 (默认: 5)')

    args = parser.parse_args()

    print("="*70)
    print("Costco 在线学习预测")
    print("="*70)
    print(f"\n数据集: {args.dataset}")
    print(f"初始窗口: {args.initial_window}")
    print(f"滑动窗口: {args.window_size}")
    print(f"隐藏层大小: {args.hidden_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")

    # 构建矩阵路径
    matrix_path = os.path.join(args.matrix_dir, f'{args.dataset}_matrix.npy')

    if not os.path.exists(matrix_path):
        print(f"\n✗ 错误：找不到矩阵文件 {matrix_path}")
        print(f"请先在 handleData 目录运行转换脚本:")
        print(f"  cd handleData")
        print(f"  python convert_robust.py --dataset {args.dataset} --time_axis col")
        return

    # 创建在线预测器
    predictor = OnlinePredictor(
        matrix_path=matrix_path,
        initial_window=args.initial_window,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        epochs=args.epochs
    )

    # 执行在线学习
    predictions, true_values = predictor.online_predict()

    # 保存结果
    predictor.save_results(predictions, true_values, args.output_dir)

    print(f"\n{'='*70}")
    print("✓ 在线学习完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
