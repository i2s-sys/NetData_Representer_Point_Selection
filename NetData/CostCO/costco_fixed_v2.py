"""
CostCo 矩阵补全模型 - 全量训练版本 + Loss 曲线图
修改点：
1. 训练次数改为 20 次
2. 记录每次训练的 loss
3. 绘制训练 loss 曲线图
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class CostCo_Matrix(nn.Module):
    """
    CostCo 模型 - 2 个卷积版本
    两个嵌入层（链路+时间）+ 两个卷积层
    """
    def __init__(self, num_routes, num_time, embedding_dim, nc=100):
        super(CostCo_Matrix, self).__init__()
        
        # 两个嵌入层
        self.route_embeddings = nn.Embedding(num_routes, embedding_dim)
        self.time_embeddings = nn.Embedding(num_time, embedding_dim)

        # 第1个卷积层：在嵌入维度上
        self.conv1 = nn.Conv2d(1, nc, (1, embedding_dim), padding=0)

        # 第2个卷积层：在模态维度上
        self.conv2 = nn.Conv2d(nc, nc, (2, 1), padding=0)

        # 全连接层
        self.fc1 = nn.Linear(nc, 1)
    
    def forward(self, route_idx, time_idx):
        batch_size = route_idx.size(0)
        
        # 获取嵌入
        route_embeds = self.route_embeddings(route_idx)
        time_embeds = self.time_embeddings(time_idx)
        
        # 拼接
        H = torch.cat((route_embeds.unsqueeze(1), time_embeds.unsqueeze(1)), 1)
        H = H.unsqueeze(1)
        
        # 卷积
        x = torch.relu(self.conv1(H))
        x = torch.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        
        return x


# ==================== 自定义损失函数 ====================
class CustomLoss(nn.Module):
    def __init__(self, loss_type='mae'):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, y_pred, y_true):
        if self.loss_type == 'mae':
            loss = torch.abs(y_pred - y_true)
        elif self.loss_type == 'mse':
            loss = (y_pred - y_true) ** 2
        elif self.loss_type == 'mae_mse':
            loss = torch.abs(y_pred - y_true) + 0.5 * (y_pred - y_true) ** 2
        else:
            raise ValueError(f"未知的损失类型: {self.loss_type}")
        return loss


# ==================== 随机采样方法（全量数据版本）====================
def random_sampling_full_matrix(matrix_data, seed_num, sample_rate=0.8, min_train_samples=100):
    """
    从完整矩阵中随机采样
    不使用滑动窗口
    """
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    
    num_routes, num_time = matrix_data.shape
    total_elements = num_routes * num_time
    
    # 计算训练样本数
    num_train = int(total_elements * sample_rate)
    num_train = max(num_train, min_train_samples)
    
    # 生成随机索引
    train_indices = np.random.choice(total_elements, num_train, replace=False)
    mask = np.zeros(total_elements, dtype=bool)
    mask[train_indices] = True
    
    keep_mask = mask.reshape(matrix_data.shape)
    missing_mask = ~keep_mask
    
    # 构建训练样本
    train_samples = []
    
    for r in range(num_routes):
        for t in range(num_time):
            val = matrix_data[r, t]
            if not np.isnan(val) and val != 0:
                if keep_mask[r, t]:
                    train_samples.append((r, t, val))
    
    return train_samples


# ==================== 在线学习类 ====================
class OnlineCostCoLearner:
    """
    CostCo 在线学习器 - 全量数据版本 + Loss 曲线图
    使用完整的历史数据作为训练数据，不使用滑动窗口
    """
    def __init__(self, matrix_data, config):
        self.matrix_data = matrix_data
        self.num_routes, self.num_time = matrix_data.shape
        
        # 配置参数
        self.embedding_dim = config.get('embedding_dim', 64)
        self.nc = config.get('nc', 128)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-7)
        self.epochs_per_step = config.get('epochs_per_step', 150)
        self.history_start = config.get('history_start', 0)
        self.history_end = config.get('history_end', 2000)
        self.save_dir = config.get('save_dir', './online_results_full_matrix')
        self.sample_rate = config.get('sample_rate', 0.8)
        self.loss_type = config.get('loss_type', 'mae')
        self.global_seed = config.get('global_seed', 42)
        self.min_train_samples = config.get('min_train_samples', 100)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.predictions = []
        self.ground_truth = []
        self.prediction_errors = []
        
        self.model = None
        self.history = []
        self.training_data = None  # 只准备一次
    
    def set_seed(self, seed=None):
        """设置固定随机种子"""
        if seed is None:
            seed = self.global_seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def create_model(self):
        """创建模型"""
        model = CostCo_Matrix(
            num_routes=self.num_routes,
            num_time=self.num_time,
            embedding_dim=self.embedding_dim,
            nc=self.nc
        ).to(device)
        return model
    
    def prepare_training_data(self):
        """
        准备训练数据 - 使用完整的历史数据
        """
        # 使用完整的历史数据 [0, 1999]
        history_data = self.matrix_data[:, :self.history_end]
        
        # 从完整历史数据中随机采样训练样本
        train_samples = random_sampling_full_matrix(
            history_data,
            seed_num=self.global_seed,
            sample_rate=self.sample_rate,
            min_train_samples=self.min_train_samples
        )
        
        if len(train_samples) == 0:
            print(f"  错误: 没有可用的训练样本")
            return None, None, None
        
        # 转换为 tensor
        route_indices = torch.tensor([s[0] for s in train_samples], dtype=torch.long).to(device)
        time_indices = torch.tensor([s[1] for s in train_samples], dtype=torch.long).to(device)
        values = torch.tensor([s[2] for s in train_samples], dtype=torch.float32).to(device).unsqueeze(1)
        
        return route_indices, time_indices, values
    
    def train_model(self, epochs=None, verbose=True):
        """
        训练模型 - 记录每个 epoch 的 loss
        返回: (平均 loss, 每个 epoch 的 loss 列表)
        """
        if epochs is None:
            epochs = self.epochs_per_step
        
        # 只在初始训练时准备数据（只准备一次）
        if self.training_data is None:
            print("  准备训练数据")
            route_indices, time_indices, values = self.prepare_training_data()
            self.training_data = (route_indices, time_indices, values)
        
        route_indices, time_indices, values = self.training_data
        
        if values is None:
            return None, None
        
        batch_size = min(128, len(values))
        if batch_size < 10:
            print(f"  警告: batch_size 较小 ({batch_size})，使用完整批训练")
            batch_size = len(values)
        
        dataset = torch.utils.data.TensorDataset(route_indices, time_indices, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 选择损失函数
        if self.loss_type == 'mae':
            criterion = CustomLoss('mae')
        elif self.loss_type == 'mse':
            criterion = CustomLoss('mse')
        elif self.loss_type == 'mae_mse':
            criterion = CustomLoss('mae_mse')
        
        # 记录每个 epoch 的 loss
        epoch_losses = []
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_routes, batch_times, batch_values in dataloader:
                predictions = self.model(batch_routes, batch_times)
                loss = criterion(predictions, batch_values)
                
                total_loss += loss.sum().item()
                
                loss.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            avg_loss = total_loss / len(values)
            epoch_losses.append(avg_loss)
            
            if verbose:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Samples: {len(values)}, Batch: {batch_size}")
        
        return avg_loss, epoch_losses
    
    def predict_next(self, target_time_idx):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            route_indices = torch.arange(self.num_routes, dtype=torch.long).to(device)
            time_indices = torch.full((self.num_routes,), target_time_idx, dtype=torch.long).to(device)
            predictions = self.model(route_indices, time_indices)
        
        return predictions.cpu().numpy().flatten()
    
    def run_online_learning(self):
        """运行在线学习"""
        print("="*80)
        print("CostCo 在线学习 - 全量数据版本 + Loss 曲线图")
        print("="*80)
        print(f"数据形状: {self.matrix_data.shape}")
        print(f"历史数据: [0, {self.history_end - 1}]")
        print(f"预测范围: [{self.history_end}, {self.num_time - 1}]")
        print(f"采样率: {self.sample_rate}")
        print(f"训练轮数: {self.epochs_per_step}")
        print(f"随机种子: {self.global_seed}")
        print("="*80 + "\n")
        
        self.set_seed(self.global_seed)
        
        # ==================== 阶段1: 训练模型 ====================
        print("\n【阶段1】训练模型")
        print(f"使用完整的历史数据 [0, {self.history_end - 1}] 训练模型")
        print(f"训练轮数: {self.epochs_per_step}")
        self.current_time = self.history_end
        
        # 创建模型
        self.model = self.create_model()
        print(f"\n模型结构:")
        print(self.model)
        print(f"\n模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 训练
        print("开始训练...")
        final_avg_loss, epoch_losses = self.train_model(epochs=self.epochs_per_step, verbose=True)
        print(f"\n训练完成！")
        print(f"最终平均损失: {final_avg_loss:.6f}")
        print(f"每个 epoch 的 loss: {[f'{l:.4f}' for l in epoch_losses]}")
        
        # 保存模型
        self.save_model('trained_model.pth')
        
        # ==================== 阶段2: 绘制 Loss 曲线图 ====================
        print("\n绘制训练 Loss 曲线图...")
        self.plot_training_loss_curve(epoch_losses)
        
        # ==================== 阶段3: 预测 ====================
        print(f"\n【阶段2】预测")
        print(f"使用训练好的模型预测时间点 {self.history_end} 到 {self.num_time - 1}")
        print(f"将预测 {self.num_time - self.history_end} 个时间点")
        print(f"直接预测，不再训练")
        print("="*80 + "\n")
        
        prediction_times = list(range(self.history_end, self.num_time))
        
        for pred_time in prediction_times:
            print(f"\n>>> 预测时间点 {pred_time}/{self.num_time - 1}")
            print(f"  训练数据: 完整的历史数据 [0, {self.history_end - 1}]")
            print(f"  训练样本数: {len(self.training_data[0])}")
            print(f"  使用的随机种子: {self.global_seed} (固定）")
            
            # 直接预测，不训练
            predictions = self.predict_next(pred_time)
            
            # 评估
            ground_truth = self.matrix_data[:, pred_time]
            
            valid_mask = ~np.isnan(ground_truth) & (ground_truth != 0)
            if valid_mask.any():
                valid_predictions = predictions[valid_mask]
                valid_ground_truth = ground_truth[valid_mask]
                
                mae = np.mean(np.abs(valid_predictions - valid_ground_truth))
                mse = np.mean((valid_predictions - valid_ground_truth) ** 2)
                rmse = np.sqrt(mse)
                
                print(f"  预测完成!")
                print(f"  预测MAE: {mae:.6f}")
                print(f"  预测MSE: {mse:.6f}")
                print(f"  预测RMSE: {rmse:.6f}")
                print(f"  有效预测数: {valid_mask.sum()}/{len(valid_mask)}")
                
                self.predictions.append(predictions)
                self.ground_truth.append(ground_truth)
                self.prediction_errors.append({
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse
                })
                self.history.append({
                    'time': pred_time,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse
                })
            else:
                print(f"  警告: 时间点 {pred_time} 没有有效的真实值")
                self.predictions.append(predictions)
                self.ground_truth.append(ground_truth)
            
            # 每100个时间点保存一次模型
            if (pred_time - self.history_end) % 100 == 0 and pred_time > self.history_end:
                checkpoint_name = f'checkpoint_time_{pred_time}.pth'
                self.save_model(checkpoint_name)
                print(f"  模型检查点已保存: {checkpoint_name}")
        
        print("\n" + "="*80)
        print("预测完成!")
        print("="*80)
        
        self.save_results()
        self.compute_overall_metrics()
        self.save_config()
    
    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"  模型已保存: {model_path}")
    
    def save_results(self):
        """保存结果"""
        predictions_array = np.array(self.predictions)
        ground_truth_array = np.array(self.groud_truth)
        
        pred_file = os.path.join(self.save_dir, 'predictions.npy')
        np.save(pred_file, predictions_array)
        print(f"\n预测结果已保存: {pred_file}")
        
        gt_file = os.path.join(self.save_dir, 'ground_truth.npy')
        np.save(gt_file, ground_truth_array)
        print(f"真实值已保存: {gt_file}")
        
        history_file = os.path.join(self.save_dir, 'prediction_history.npy')
        np.save(history_file, np.array(self.history))
        print(f"预测历史已保存: {history_file}")
    
    def compute_overall_metrics(self):
        """计算总体评估指标"""
        print("\n" + "="*80)
        print("总体评估指标")
        print("="*80)
        
        all_mae = [h['mae'] for h in self.history if 'mae' in h]
        all_mse = [h['mse'] for h in self.history if 'mse' in h]
        all_rmse = [h['rmse'] for h in self.history if 'rmse' in h]
        
        if all_mae:
            print(f"平均预测MAE: {np.mean(all_mae):.6f}")
            print(f"平均预测MSE: {np.mean(all_mse):.6f}")
            print(f"平均预测RMSE: {np.mean(all_rmse):.6f}")
            print(f"MAE标准差: {np.std(all_mae):.6f}")
            print(f"最小MAE: {np.min(all_mae):.6f}")
            print(f"最大MAE: {np.max(all_mae):.6f}")
    
    def save_config(self):
        """保存配置"""
        config_data = {
            'global_seed': self.global_seed,
            'embedding_dim': self.embedding_dim,
            'nc': self.nc,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'epochs_per_step': self.epochs_per_step,
            'sample_rate': self.sample_rate,
            'min_train_samples': self.min_train_samples,
            'loss_type': self.loss_type,
            'history_start': self.history_start,
            'history_end': self.history_end,
            'matrix_shape': self.matrix_data.shape,
            'training_mode': 'full_matrix',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        config_file = os.path.join(self.save_dir, 'config_and_seed.txt')
        with open(config_file, 'w') as f:
            f.write("模型配置和随机种子信息\n")
            f.write("="*50 + "\n")
            for key, value in config_data.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "="*50)
            f.write(f"\n关键特性:\n")
            f.write(f"- 使用完整的历史矩阵作为训练数据\n")
            f.write(f"- 不使用滑动窗口\n")
            f.write(f"- 训练轮数: {self.epochs_per_step}\n")
            f.write(f"- 记录每个 epoch 的 loss\n")
            f.write(f"- 绘制训练 loss 曲线图\n")
            f.write(f"- 固定随机种子: {self.global_seed}\n")
        
        print(f"\n配置和种子信息已保存: {config_file}")
    
    def plot_training_loss_curve(self, epoch_losses):
        """绘制训练 loss 曲线图"""
        print("  正在绘制 Loss 曲线图...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(1, len(epoch_losses) + 1)
        
        # 绘制 loss 曲线
        ax.plot(epochs, epoch_losses, 'b-', linewidth=2, label='Training Loss')
        ax.fill_between(epochs, epoch_losses, color='blue', alpha=0.1)
        
        # 标记最终 loss
        ax.scatter([len(epoch_losses)], [epoch_losses[-1]], 
                   color='red', s=100, zorder=5, label='Final Loss')
        ax.annotate(f'{epoch_losses[-1]:.4f}', 
                   xy=(len(epoch_losses), epoch_losses[-1]),
                   xytext=(len(epoch_losses) + 0.5, epoch_losses[-1] + max(epoch_losses) * 0.02),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(self.save_dir, 'training_loss_curve.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Loss 曲线图已保存: {plot_file}")
        plt.close()


def main():
    """主函数"""
    matrix_file = './output/Geant_23_23_3000_matrix_col_time.npy'
    
    print(f"加载数据文件: {matrix_file}")
    if not os.path.exists(matrix_file):
        print(f"错误: 数据文件不存在: {matrix_file}")
        print(f"请确保已运行 convert_to_matrix_col_time.py 生成矩阵文件")
        return
    
    matrix_data = np.load(matrix_file)
    print(f"数据加载成功，形状: {matrix_data.shape}")
    print(f"数据类型: {matrix_data.dtype}")
    print(f"数据范围: [{np.nanmin(matrix_data):.2f}, {np.nanmax(matrix_data):.2f}]")
    
    config = {
        # 模型参数
        'embedding_dim': 64,
        'nc': 128,
        
        # 训练参数
        'lr': 1e-4,
        'weight_decay': 1e-7,
        'epochs_per_step': 20,  # 修改：训练 20 次
        'loss_type': 'mae',
        
        # 采样配置
        'sample_rate': 0.8,
        'global_seed': 42,
        'min_train_samples': 100,
        
        # 数据划分
        'history_start': 0,
        'history_end': 2000,
        
        # 保存
        'save_dir': './online_results_full_matrix'
    }
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n关键特性:")
    print("  - 使用完整的历史矩阵作为训练数据")
    print("  - 不使用滑动窗口")
    print(f"  - 训练轮数: {config['epochs_per_step']} 次（记录每次 loss）")
    print("  - 绘制训练 loss 曲线图")
    print("  - 只训练一次，然后预测所有时间点")
    print("  - 固定随机种子，结果可重复")
    
    learner = OnlineCostCoLearner(matrix_data, config)
    learner.run_online_learning()
    
    print("\n程序执行完成!")
    print("提示：")
    print("  - 训练 loss 曲线图已保存在 online_results_full_matrix 目录")
    print("  - 最终模型已保存为 trained_model.pth")


if __name__ == '__main__':
    main()
