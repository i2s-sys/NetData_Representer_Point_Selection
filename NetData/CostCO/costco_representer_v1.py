"""
CostCo 矩阵补全模型 - 代表点采样版本 这个版本是训练好之后全量测试的
修复维度越界错误：route_grad[r] -> route_grad[r, :]
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


# ==================== CostCo 模型定义（2 卷积版本）====================
class CostCo_Matrix(nn.Module):
    """
    CostCo 模型 - 2 个卷积版本
    """
    def __init__(self, num_routes, num_time, embedding_dim, nc=100):
        super(CostCo_Matrix, self).__init__()
        
        # 两个嵌入层
        self.route_embeddings = nn.Embedding(num_routes, embedding_dim)
        self.time_embeddings = nn.Embedding(num_time, embedding_dim)

        # 两个卷积层
        self.conv1 = nn.Conv2d(1, nc, (1, embedding_dim), padding=0)
        self.conv2 = nn.Conv2d(nc, nc, (2, 1), padding=0)

        # 全连接层
        self.fc1 = nn.Linear(nc, 1)
    
    def forward(self, route_idx, time_idx):
        batch_size = route_idx.size(0)
        
        # 确保索引是正确的形状
        if route_idx.dim() == 0:
            route_idx = route_idx.view(-1)  # 转换为 1D
        time_idx = time_idx.view(-1)
        
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


# ==================== 代表点重要性计算（修复版）====================
def compute_sample_importance_gradient(model, route_idx, time_idx, criterion, values):
    """
    计算每个样本的重要性（基于梯度）
    修复：使用切片避免索引越界
    """
    batch_size = route_idx.size(0)
    
    # 前向传播
    model.eval()
    with torch.enable_grad():
        predictions = model(route_idx, time_idx)
        loss = criterion(predictions, values)
        
        # 计算损失
        total_loss = loss.sum()
        
        # 反向传播
        total_loss.backward()
    
    # 获取嵌入层梯度（样本的贡献）
    # 修复：使用切片而不是直接索引
    route_grad = model.route_embeddings.weight.grad.abs()  # [num_routes, embedding_dim]
    time_grad = model.time_embeddings.weight.grad.abs()  # [num_time, embedding_dim]
    
    # 计算每个样本的重要性
    sample_importances = []
    
    for i in range(batch_size):
        r = route_idx[i].item()
        t = time_idx[i].item()

        route_imp = torch.sum(route_grad[r, :] ** 2).item()
        time_imp = torch.sum(time_grad[t, :] ** 2).item()
        
        # 样本重要性 = 链路贡献 + 时间贡献
        imp = route_imp + time_imp
        sample_importances.append(imp)
    
    sample_importances = np.array(sample_importances)
    
    # 归一化
    sample_importances = sample_importances / (sample_importances.mean() + 1e-8)
    
    grad_info = {
        'route_grad_norm': route_grad.norm().item(),
        'time_grad_norm': time_grad.norm().item(),
        'mean_importance': sample_importances.mean(),
        'std_importance': sample_importances.std(),
        'min_importance': sample_importances.min(),
        'max_importance': sample_importances.max()
    }
    
    return sample_importances, grad_info


# ==================== 随机采样方法（支持代表点）====================
def random_sampling_with_representer(matrix_data, seed_num, sample_rate=0.8, min_train_samples=100, use_representer=False):
    """
    随机采样（支持代表点）
    """
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    
    num_routes, num_time = matrix_data.shape
    total_elements = num_routes * num_time
    
    num_train = int(total_elements * sample_rate)
    num_train = max(num_train, min_train_samples)
    
    # 生成随机索引
    train_indices = np.random.choice(total_elements, num_train, replace=False)
    mask = np.zeros(total_elements, dtype=bool)
    mask[train_indices] = True
    
    keep_mask = mask.reshape(matrix_data.shape)
    
    # 构建训练样本
    train_samples = []
    
    for r in range(num_routes):
        for t in range(num_time):
            val = matrix_data[r, t]
            if not np.isnan(val) and val != 0:
                if keep_mask[r, t]:
                    train_samples.append((r, t, val))
    
    return train_samples


# ==================== 在线学习类（代表点版本）====================
class OnlineCostCoLearner:
    """
    CostCo 在线学习器 - 代表点采样版本
    """
    def __init__(self, matrix_data, config):
        self.matrix_data = matrix_data
        self.num_routes, self.num_time = matrix_data.shape
        
        # 配置
        self.embedding_dim = config.get('embedding_dim', 64)
        self.nc = config.get('nc', 128)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-7)
        self.epochs_per_step = config.get('epochs_per_step', 50)
        self.history_start = config.get('history_start', 0)
        self.history_end = config.get('history_end', 2000)
        self.save_dir = config.get('save_dir', './online_results_representer')
        self.sample_rate = config.get('sample_rate', 0.8)
        self.loss_type = config.get('loss_type', 'mae')
        self.global_seed = config.get('global_seed', 42)
        self.min_train_samples = config.get('min_train_samples', 100)
        self.use_representer = config.get('use_representer', True)
        self.representer_method = config.get('representer_method', 'gradient')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.predictions = []
        self.ground_truth = []
        self.prediction_errors = []
        
        self.model = None
        self.history = []
        self.training_data = None
        self.sample_importances = None
    
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
        """准备训练数据"""
        history_data = self.matrix_data[:, :self.history_end]
        
        train_samples = random_sampling_with_representer(
            history_data,
            seed_num=self.global_seed,
            sample_rate=self.sample_rate,
            min_train_samples=self.min_train_samples,
            use_representer=False  # 初始训练不使用代表点
        )
        
        if len(train_samples) == 0:
            print(f"  错误: 没有可用的训练样本")
            return None, None, None
        
        # 转换为 tensor，确保是 1D
        route_indices = torch.tensor([s[0] for s in train_samples], dtype=torch.long).to(device)
        time_indices = torch.tensor([s[1] for s in train_samples], dtype=torch.long).to(device)
        values = torch.tensor([s[2] for s in train_samples], dtype=torch.float32).to(device).unsqueeze(1)
        
        # 确保索引形状正确
        if route_indices.dim() == 0:
            route_indices = route_indices.view(-1)
        if time_indices.dim() == 0:
            time_indices = time_indices.view(-1)
        
        print(f"  训练样本数: {len(train_samples)}")
        print(f"  训练样本比例: {len(train_samples) / (self.num_routes * self.history_end):.2%}")
        
        return route_indices, time_indices, values
    
    def train_model_with_representer(self, epochs=None, verbose=True):
        """
        训练模型 - 使用代表点重要性
        """
        if epochs is None:
            epochs = self.epochs_per_step
        
        # 获取训练数据
        if self.training_data is None:
            route_indices, time_indices, values = self.prepare_training_data()
            self.training_data = (route_indices, time_indices, values)
        
        route_indices, time_indices, values = self.training_data
        
        if values is None:
            return None, None, None
        
        batch_size = min(128, len(values))
        if batch_size < 10:
            batch_size = len(values)
        
        dataset = torch.utils.data.TensorDataset(route_indices, time_indices, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if self.loss_type == 'mae':
            criterion = CustomLoss('mae')
        elif self.loss_type == 'mse':
            criterion = CustomLoss('mse')
        elif self.loss_type == 'mae_mse':
            criterion = CustomLoss('mae_mse')
        
        # 如果使用代表点，计算重要性
        if self.use_representer:
            print("  计算样本重要性...")
            sample_importances, grad_info = compute_sample_importance_gradient(
                self.model,
                route_indices,
                time_indices,
                criterion,
                values
            )
            self.sample_importances = sample_importances
            
            print(f"  样本重要性统计:")
            print(f"    平均: {grad_info['mean_importance']:.6f}")
            print(f"    标准差: {grad_info['std_importance']:.6f}")
            print(f"    最小: {grad_info['min_importance']:.6f}")
            print(f"    最大: {grad_info['max_importance']:.6f}")
            print(f"    链路梯度范数: {grad_info['route_grad_norm']:.6f}")
            print(f"    时间梯度范数: {grad_info['time_grad_norm']:.6f}")
        
        epoch_importances = []
        
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
            
            if verbose:
                if self.use_representer and sample_importances is not None:
                    print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Samples: {len(values)}, Batch: {batch_size}, Avg Imp: {np.mean(sample_importances):.6f}")
                else:
                    print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Samples: {len(values)}, Batch: {batch_size}")
        
        return avg_loss, epoch_importances
    
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
        print("CostCo 在线学习 - 代表点采样版本（修复索引错误）")
        print("="*80)
        print(f"数据形状: {self.matrix_data.shape}")
        print(f"历史数据: [0, {self.history_end - 1}]")
        print(f"预测范围: [{self.history_end}, {self.num_time - 1}]")
        print(f"采样率: {self.sample_rate}")
        print(f"使用代表点: {self.use_representer}")
        print(f"代表点方法: {self.representer_method}")
        print("="*80 + "\n")
        
        self.set_seed(self.global_seed)
        
        # 阶段1: 训练
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
        final_avg_loss, epoch_importances = self.train_model_with_representer(epochs=self.epochs_per_step, verbose=True)
        print(f"\n训练完成！")
        print(f"最终平均损失: {final_avg_loss:.6f}")
        
        self.save_model('trained_model.pth')
        
        # 保存样本重要性
        if self.sample_importances is not None:
            importance_file = os.path.join(self.save_dir, 'sample_importances.npy')
            np.save(importance_file, self.sample_importances)
            print(f"样本重要性已保存: {importance_file}")
        
        # 绘制重要性分布
        if self.sample_importances is not None:
            self.plot_importance_distribution(self.sample_importances)
        
        # 阶段2: 预测
        print(f"\n【阶段2】预测")
        print(f"使用训练好的模型预测时间点 {self.history_end} 到 {self.num_time - 1}")
        print("="*80 + "\n")
        
        prediction_times = list(range(self.history_end, self.num_time))
        
        for pred_time in prediction_times:
            print(f"\n>>> 预测时间点 {pred_time}/{self.num_time - 1}")
            print(f"  训练数据: 完整的历史数据 [0, {self.history_end - 1}]")
            print(f"  训练样本数: {len(self.training_data[0])}")
            
            # 直接预测
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
        
        print("\n" + "="*80)
        print("在线学习完成!")
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
        ground_truth_array = np.array(self.ground_truth)
        
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
            'use_representer': self.use_representer,
            'representer_method': self.representer_method,
            'history_start': self.history_start,
            'history_end': self.history_end,
            'matrix_shape': self.matrix_data.shape,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        config_file = os.path.join(self.save_dir, 'config_and_seed.txt')
        with open(config_file, 'w') as f:
            f.write("模型配置和随机种子信息\n")
            f.write("="*50 + "\n")
            for key, value in config_data.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "="*50)
            f.write(f"\n修复说明:\n")
            f.write(f"- 修复了索引越界错误: route_grad[r] -> route_grad[r, :]\n")
            f.write(f"- 使用切片确保正确的维度\n")
            f.write(f"- 重要性 = 梯度 L2 范数（链路 + 时间）\n")
        
        print(f"\n配置和种子信息已保存: {config_file}")
    
    def plot_importance_distribution(self, sample_importances):
        """绘制样本重要性分布"""
        print("  正在绘制样本重要性分布图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 直方图
        axes[0, 0].hist(sample_importances, bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('样本重要性分布（直方图）', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('重要性', fontsize=10)
        axes[0, 0].set_ylabel('样本数', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 排序
        sorted_importances = np.sort(sample_importances)[::-1]
        axes[0, 1].plot(range(len(sorted_importances)), sorted_importances, 'b-', linewidth=2)
        axes[0, 1].set_title('样本重要性排序', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('样本索引（按重要性排序）', fontsize=10)
        axes[0, 1].set_ylabel('重要性', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 累积
        axes[1, 0].plot(np.arange(1, len(sample_importances) + 1), 
                     np.cumsum(np.sort(sample_importances)[::-1]), 'b-', linewidth=2)
        axes[1, 0].set_title('累计重要性分布', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('样本数', fontsize=10)
        axes[1, 0].set_ylabel('累计重要性', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 箱线图
        axes[1, 1].boxplot(sample_importances, patch_artist=None)
        axes[1, 1].set_title('样本重要性箱线图', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('重要性', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(self.save_dir, 'sample_importance_distribution.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  样本重要性分布图已保存: {plot_file}")
        plt.close()


def main():
    """主函数"""
    matrix_file = './output/Geant_23_23_3000_matrix_col_time.npy'
    
    print(f"加载数据文件: {matrix_file}")
    if not os.path.exists(matrix_file):
        print(f"错误: 数据文件不存在: {matrix_file}")
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
        'epochs_per_step': 50,
        'loss_type': 'mae',
        
        # 代表点配置
        'use_representer': True,  # 使用代表点
        'representer_method': 'gradient',  # 梯度法
        
        # 采样配置
        'sample_rate': 0.8,
        'global_seed': 42,
        'min_train_samples': 100,
        
        # 数据划分
        'history_start': 0,
        'history_end': 2000,
        
        # 保存
        'save_dir': './online_results_representer'
    }
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n关键特性:")
    print("  - 修复了索引越界错误")
    print("  - 使用切片: route_grad[r, :] 而不是 route_grad[r]")
    print("  - 计算每个不缺失样本的重要性")
    print("  - 重要性 = 梯度 L2 范数（链路 + 时间）")
    print("  - 可视化样本重要性分布（4 个子图）")
    
    learner = OnlineCostCoLearner(matrix_data, config)
    learner.run_online_learning()
    
    print("\n程序执行完成!")
    print("提示：")
    print("  - sample_importances.npy: 每个样本的重要性")
    print("  - sample_importance_distribution.png: 4 个子图")


if __name__ == '__main__':
    main()
