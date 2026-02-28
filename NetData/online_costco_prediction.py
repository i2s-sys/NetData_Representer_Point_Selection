"""
CostCo 在线学习模型 - 逐步预测未来数据
使用历史数据逐步训练，预测每个时间点的未来值
修复版本：使用3个嵌入层（i, j, k），完全匹配原始CostCo实现
"""

import torch as t
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ==================== CostCo 模型定义（完全匹配原始实现）====================
class CostCo(nn.Module):
    """
    CostCo: A Neural Tensor Completion Model for Sparse Tensors
    论文: CoSTCo (KDD 2018)
    完全匹配原始实现：使用3个嵌入层（i, j, k）
    """
    def __init__(self, i_size, j_size, k_size, embedding_dim, nc=100):
        super(CostCo, self).__init__()
        # 三个嵌入层：i（源节点）、j（目标节点）、k（时间）
        self.iembeddings = nn.Embedding(i_size, embedding_dim)
        self.jembeddings = nn.Embedding(j_size, embedding_dim)
        self.kembeddings = nn.Embedding(k_size, embedding_dim)

        # 两个卷积层
        self.conv1 = nn.Conv2d(1, nc, (1, embedding_dim))
        self.conv2 = nn.Conv2d(nc, nc, (3, 1))

        # 全连接层
        self.fc1 = nn.Linear(nc, 1)

    def forward(self, i_input, j_input, k_input):
        """
        前向传播 - 完全匹配原始实现
        Args:
            i_input: 源节点索引 [batch_size]
            j_input: 目标节点索引 [batch_size]
            k_input: 时间索引 [batch_size]
        Returns:
            预测值 [batch_size, 1]
        """
        # embedding
        lookup_itensor = i_input.long()
        lookup_jtensor = j_input.long()
        lookup_ktensor = k_input.long()

        i_embeds = self.iembeddings(lookup_itensor).unsqueeze(1)  # [batch, 1, embedding_dim]
        j_embeds = self.jembeddings(lookup_jtensor).unsqueeze(1)  # [batch, 1, embedding_dim]
        k_embeds = self.kembeddings(lookup_ktensor).unsqueeze(1)  # [batch, 1, embedding_dim]

        # 拼接三个嵌入向量
        H = t.cat((i_embeds, j_embeds, k_embeds), 2)  # [batch, 3, embedding_dim]
        # H = H.unsqueeze(1)  # 不需要这一行，cat已经在维度2上拼接

        # 卷积操作
        x = t.relu(self.conv1(H))
        x = t.relu(self.conv2(x))
        x = x.view(-1, x.shape[1])  # [batch, nc]
        x = self.fc1(x)

        return x


# ==================== 在线学习类 ====================
class OnlineCostCoLearner:
    """
    CostCo 在线学习器
    实现增量学习：逐步训练，逐步预测
    """
    def __init__(self, matrix_data, config):
        """
        Args:
            matrix_data: 形状为 [num_routes, num_time] 的矩阵
                        每行代表一条网络链路（源节点→目标节点）
                        每列代表一个时间点的网络状态
            config: 配置字典
        """
        self.matrix_data = matrix_data  # [num_routes, num_time]
        self.num_routes, self.num_time = matrix_data.shape
        
        # 配置参数
        self.embedding_dim = config.get('embedding_dim', 50)
        self.nc = config.get('nc', 100)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-7)
        self.epochs_per_step = config.get('epochs_per_step', 10)
        self.history_start = config.get('history_start', 0)
        self.history_end = config.get('history_end', 2000)
        self.save_dir = config.get('save_dir', './online_results')
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 存储预测结果和真实值
        self.predictions = []
        self.ground_truth = []
        self.prediction_errors = []
        
        # 模型
        self.model = None
        
        # 记录信息
        self.history = []
    
    def prepare_training_data(self, end_time_idx):
        """
        准备训练数据：从history_start到end_time_idx的所有时间点
        将矩阵数据转换为 CostCo 需要的 (i, j, k, value) 格式
        Args:
            end_time_idx: 训练数据的结束时间索引（包含）
        Returns:
            i_indices: [N, 1] 源节点索引
            j_indices: [N, 1] 目标节点索引
            k_indices: [N, 1] 时间索引
            values: [N, 1] 对应值
        """
        i_indices = []
        j_indices = []
        k_indices = []
        values = []
        
        start_time = self.history_start
        end_time = min(end_time_idx, self.num_time - 1)
        
        for t in range(start_time, end_time + 1):
            for r in range(self.num_routes):
                val = self.matrix_data[r, t]
                if not np.isnan(val) and val != 0:  # 只使用有效数据
                    i_indices.append(0)  # 源节点固定为0（简化）
                    j_indices.append(r)  # 目标节点是链路索引
                    k_indices.append(t)  # 时间索引
                    values.append(val)
        
        # 转换为tensor
        i_indices = torch.tensor(i_indices, dtype=torch.long).to(device)
        j_indices = torch.tensor(j_indices, dtype=torch.long).to(device)
        k_indices = torch.tensor(k_indices, dtype=torch.long).to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device).unsqueeze(1)
        
        return i_indices, j_indices, k_indices, values
    
    def train_model(self, epochs=None, verbose=True):
        """
        训练模型
        Args:
            epochs: 训练轮数，如果为None则使用默认值
            verbose: 是否打印训练信息
        """
        if epochs is None:
            epochs = self.epochs_per_step
        
        # 准备训练数据
        i_indices, j_indices, k_indices, values = self.prepare_training_data(self.current_time - 1)
        
        if len(values) == 0:
            print(f"警告: 时间点 {self.current_time} 没有可用的训练数据")
            return
        
        # 创建DataLoader
        batch_size = min(128, len(values))
        dataset = torch.utils.data.TensorDataset(i_indices, j_indices, k_indices, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_i, batch_j, batch_k, batch_values in dataloader:
                # 前向传播
                predictions = self.model(batch_i, batch_j, batch_k)
                
                # 计算损失 (MAE)
                loss = torch.abs(predictions - batch_values)
                total_loss += loss.sum().item()
                
                # 反向传播
                loss.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            avg_loss = total_loss / len(values)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def predict_next(self, target_time_idx):
        """
        预测指定时间点的所有链路值
        Args:
            target_time_idx: 要预测的时间索引
        Returns:
            predictions: [num_routes, 1] 预测值
        """
        self.model.eval()
        with torch.no_grad():
            # 准备预测输入
            i_indices = torch.zeros(self.num_routes, dtype=torch.long).to(device)  # 源节点都为0
            j_indices = torch.arange(self.num_routes, dtype=torch.long).to(device)  # 目标节点是链路索引
            k_indices = torch.full((self.num_routes,), target_time_idx, dtype=torch.long).to(device)
            
            # 预测
            predictions = self.model(i_indices, j_indices, k_indices)
        
        return predictions.cpu().numpy().flatten()
    
    def run_online_learning(self):
        """
        运行在线学习流程
        逐步训练并预测未来时间点
        """
        print("="*80)
        print("开始 CostCo 在线学习")
        print("="*80)
        print(f"数据形状: {self.matrix_data.shape}")
        print(f"  行数（链路数）: {self.num_routes}")
        print(f"  列数（时间点）: {self.num_time}")
        print(f"历史数据范围: [{self.history_start}, {self.history_end}]")
        print(f"预测范围: [{self.history_end}, {self.num_time - 1}]")
        print(f"每步训练轮数: {self.epochs_per_step}")
        print(f"学习率: {self.lr}")
        print(f"嵌入维度: {self.embedding_dim}")
        print(f"卷积通道数: {self.nc}")
        print("="*80 + "\n")
        
        # 预测的时间点列表
        prediction_times = list(range(self.history_end, self.num_time))
        
        # 阶段1: 使用初始历史数据训练
        print("\n【阶段1】初始训练")
        print(f"使用时间 [{self.history_start}, {self.history_end - 1}] 的数据训练模型")
        print(f"训练数据范围: 前 {self.history_end} 个时间点")
        self.current_time = self.history_end
        
        # 初始化模型
        # 注意：i_size=1（源节点只有1个），j_size=num_routes（目标节点是链路数）
        self.model = CostCo(
            i_size=1,               # 源节点维度
            j_size=self.num_routes,  # 目标节点维度（链路数）
            k_size=self.num_time,     # 时间维度
            embedding_dim=self.embedding_dim,
            nc=self.nc
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # 初始训练
        print("初始训练中...")
        initial_loss = self.train_model(epochs=50, verbose=True)
        print(f"初始训练完成，最终损失: {initial_loss:.6f}")
        
        # 保存初始模型
        self.save_model('initial_model.pth')
        
        # 阶段2: 在线预测
        print(f"\n【阶段2】在线预测")
        print(f"开始预测时间点 {self.history_end} 到 {self.num_time - 1}")
        print(f"将预测 {len(prediction_times)} 个时间点")
        print("="*80 + "\n")
        
        for pred_time in prediction_times:
            print(f"\n>>> 预测时间点 {pred_time}/{self.num_time - 1}")
            print(f"当前训练数据范围: [{self.history_start}, {pred_time - 1}]")
            print(f"训练数据包含: {pred_time} 个时间点")
            
            # 步骤1: 训练模型（使用到pred_time-1的数据）
            self.current_time = pred_time
            train_loss = self.train_model(epochs=self.epochs_per_step, verbose=True)
            
            # 步骤2: 预测pred_time
            predictions = self.predict_next(pred_time)
            
            # 步骤3: 获取真实值
            ground_truth = self.matrix_data[:, pred_time]
            
            # 步骤4: 计算误差（只计算非NaN和非零的值）
            valid_mask = ~np.isnan(ground_truth) & (ground_truth != 0)
            if valid_mask.any():
                valid_predictions = predictions[valid_mask]
                valid_ground_truth = ground_truth[valid_mask]
                
                # 计算MAE
                mae = np.mean(np.abs(valid_predictions - valid_ground_truth))
                # 计算RMSE
                rmse = np.sqrt(np.mean((valid_predictions - valid_ground_truth) ** 2))
                
                print(f"  预测完成!")
                print(f"  训练损失: {train_loss:.6f}")
                print(f"  预测MAE: {mae:.6f}")
                print(f"  预测RMSE: {rmse:.6f}")
                print(f"  有效预测数: {valid_mask.sum()}/{len(valid_mask)}")
                
                # 存储结果
                self.predictions.append(predictions)
                self.ground_truth.append(ground_truth)
                self.prediction_errors.append({'mae': mae, 'rmse': rmse, 'loss': train_loss})
                self.history.append({
                    'time': pred_time,
                    'mae': mae,
                    'rmse': rmse,
                    'train_loss': train_loss
                })
            else:
                print(f"  警告: 时间点 {pred_time} 没有有效的真实值")
                self.predictions.append(predictions)
                self.ground_truth.append(ground_truth)
            
            # 每隔100个时间点保存一次模型
            if (pred_time - self.history_end) % 100 == 0 and pred_time > self.history_end:
                checkpoint_name = f'checkpoint_time_{pred_time}.pth'
                self.save_model(checkpoint_name)
                print(f"  模型检查点已保存: {checkpoint_name}")
        
        print("\n" + "="*80)
        print("在线学习完成!")
        print("="*80)
        
        # 保存结果
        self.save_results()
        
        # 计算总体统计
        self.compute_overall_metrics()
    
    def save_model(self, filename):
        """保存模型参数"""
        model_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), model_path)
    
    def save_results(self):
        """保存预测结果"""
        # 转换为numpy数组
        predictions_array = np.array(self.predictions)
        ground_truth_array = np.array(self.ground_truth)
        
        # 保存预测结果
        pred_file = os.path.join(self.save_dir, 'predictions.npy')
        np.save(pred_file, predictions_array)
        print(f"\n预测结果已保存: {pred_file}")
        
        # 保存真实值
        gt_file = os.path.join(self.save_dir, 'ground_truth.npy')
        np.save(gt_file, ground_truth_array)
        print(f"真实值已保存: {gt_file}")
        
        # 保存误差历史
        history_file = os.path.join(self.save_dir, 'prediction_history.npy')
        np.save(history_file, np.array(self.history))
        print(f"预测历史已保存: {history_file}")
    
    def compute_overall_metrics(self):
        """计算总体评估指标"""
        print("\n" + "="*80)
        print("总体评估指标")
        print("="*80)
        
        # 计算所有有效预测的MAE和RMSE
        all_mae = [h['mae'] for h in self.history if 'mae' in h]
        all_rmse = [h['rmse'] for h in self.history if 'rmse' in h]
        all_train_loss = [h['train_loss'] for h in self.history if 'train_loss' in h]
        
        if all_mae:
            print(f"平均预测MAE: {np.mean(all_mae):.6f}")
            print(f"平均预测RMSE: {np.mean(all_rmse):.6f}")
            print(f"MAE标准差: {np.std(all_mae):.6f}")
            print(f"RMSE标准差: {np.std(all_rmse):.6f}")
            print(f"最小MAE: {np.min(all_mae):.6f}")
            print(f"最大MAE: {np.max(all_mae):.6f}")
            print(f"\n平均训练损失: {np.mean(all_train_loss):.6f}")
            print(f"训练损失范围: [{np.min(all_train_loss):.6f}, {np.max(all_train_loss):.6f}]")
        
        # 绘制误差曲线
        self.plot_error_curves()
    
    def plot_error_curves(self):
        """绘制误差曲线图"""
        times = [h['time'] for h in self.history]
        mae_errors = [h['mae'] for h in self.history if 'mae' in h]
        rmse_errors = [h['rmse'] for h in self.history if 'rmse' in h]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # MAE曲线
        axes[0].plot(times, mae_errors, 'b-', linewidth=2, label='MAE')
        axes[0].set_xlabel('时间点', fontsize=12)
        axes[0].set_ylabel('MAE', fontsize=12)
        axes[0].set_title('预测误差随时间变化', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # RMSE曲线
        axes[1].plot(times, rmse_errors, 'r-', linewidth=2, label='RMSE')
        axes[1].set_xlabel('时间点', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.save_dir, 'error_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n误差曲线图已保存: {plot_file}")
        plt.close()


# ==================== 主函数 ====================
def main():
    """
    主函数：执行在线学习
    """
    # 加载数据
    matrix_file = './output/Geant_23_23_3000_matrix_col_time.npy'
    print(f"加载数据文件: {matrix_file}")
    
    if not os.path.exists(matrix_file):
        print(f"错误: 数据文件不存在: {matrix_file}")
        print(f"请确保已运行 convert_to_matrix_col_time.py 生成矩阵文件")
        return
    
    matrix_data = np.load(matrix_file)
    print(f"数据加载成功，形状: {matrix_data.shape}")
    print(f"  数据类型: {matrix_data.dtype}")
    print(f"  数据范围: [{np.nanmin(matrix_data):.2f}, {np.nanmax(matrix_data):.2f}]")
    print(f"  有效数据比例: {(~np.isnan(matrix_data) & (matrix_data != 0)).sum() / matrix_data.size * 100:.2f}%")
    
    # 配置参数
    config = {
        # 模型参数
        'embedding_dim': 50,        # 嵌入维度（建议：30-100）
        'nc': 100,                  # 卷积通道数（建议：32-128）
        
        # 训练参数
        'lr': 1e-4,                 # 学习率（建议：1e-5 到 1e-3）
        'weight_decay': 1e-7,       # 权重衰减（建议：1e-8 到 1e-6）
        'epochs_per_step': 10,       # 每步训练轮数（建议：5-20）
        
        # 数据划分
        'history_start': 0,           # 历史数据开始时间
        'history_end': 2000,          # 历史数据结束时间（训练使用前2000列）
        
        # 保存设置
        'save_dir': './online_results' # 结果保存目录
    }
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建在线学习器
    learner = OnlineCostCoLearner(matrix_data, config)
    
    # 运行在线学习
    learner.run_online_learning()
    
    print("\n程序执行完成!")


if __name__ == '__main__':
    main()
