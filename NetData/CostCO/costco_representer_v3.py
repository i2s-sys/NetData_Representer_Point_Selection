"""
CostCo 矩阵补全模型 - 这个版本是 预测单个时间点的
训练模型 -> 计算样本重要性 -> 计算行重要性 -> 选择指定比例最重要的行 vs 随机相同比例行进行对比实验
说明：每个实验都会使用选定的路线重新训练模型，然后预测所有链路在时间点 2001 的值
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

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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
        self.route_selection_ratio = config.get('route_selection_ratio', 0.1)  # 选择路线的比例
        self.stage2_epochs = config.get('stage2_epochs', 20)  # 阶段2（实验阶段）训练的epoch数
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建预测结果保存目录
        self.predictions_dir_exp1 = os.path.join(self.save_dir, 'predictions_exp1_top_routes')
        self.predictions_dir_exp2 = os.path.join(self.save_dir, 'predictions_exp2_random_routes')
        os.makedirs(self.predictions_dir_exp1, exist_ok=True)
        os.makedirs(self.predictions_dir_exp2, exist_ok=True)
        
        self.predictions = []
        self.ground_truth = []
        self.prediction_errors = []
        
        self.model = None
        self.history = []
        self.training_data = None
        self.sample_importances = None
        self.route_importances = None
        self.train_route_indices = None  # 训练样本的路线索引
    
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

        # 保存训练样本的路线索引（用于计算行重要性）
        self.train_route_indices = np.array([s[0] for s in train_samples])
        self.train_time_indices = np.array([s[1] for s in train_samples])

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
    
    def predict_routes(self, route_indices, target_time_idx):
        """预测指定路线在目标时间点的值"""
        self.model.eval()
        with torch.no_grad():
            route_tensor = torch.tensor(route_indices, dtype=torch.long).to(device)
            time_tensor = torch.full((len(route_indices),), target_time_idx, dtype=torch.long).to(device)
            predictions = self.model(route_tensor, time_tensor)

        return predictions.cpu().numpy().flatten()

    def compute_route_importance(self):
        """计算每一行（路线）的重要性"""
        if self.sample_importances is None or self.train_route_indices is None:
            print("  错误: 样本重要性或训练路线索引未计算")
            return None

        print("\n【阶段2】计算行（路线）重要性")

        # 初始化每行的重要性字典
        route_importance_sum = {}
        route_sample_count = {}

        # 遍历所有训练样本，累加到对应行
        for i, route_idx in enumerate(self.train_route_indices):
            importance = self.sample_importances[i]

            if route_idx not in route_importance_sum:
                route_importance_sum[route_idx] = 0.0
                route_sample_count[route_idx] = 0

            route_importance_sum[route_idx] += importance
            route_sample_count[route_idx] += 1

        # 计算每行的平均重要性
        route_importances = np.zeros(self.num_routes)

        for route_idx in range(self.num_routes):
            if route_idx in route_importance_sum:
                # 平均重要性 = 该行所有样本重要性之和 / 有效样本数量
                route_importances[route_idx] = route_importance_sum[route_idx] / route_sample_count[route_idx]
            else:
                # 该行没有训练样本，重要性为0
                route_importances[route_idx] = 0.0

        self.route_importances = route_importances

        # 统计信息
        print(f"  行重要性统计:")
        print(f"    平均: {route_importances.mean():.6f}")
        print(f"    标准差: {route_importances.std():.6f}")
        print(f"    最小: {route_importances.min():.6f}")
        print(f"    最大: {route_importances.max():.6f}")
        print(f"    非零行数: {(route_importances > 0).sum()}/{self.num_routes}")

        # 保存行重要性
        route_importance_file = os.path.join(self.save_dir, 'route_importances.npy')
        np.save(route_importance_file, route_importances)
        print(f"  行重要性已保存: {route_importance_file}")

        # 保存每行的样本数量
        route_count_file = os.path.join(self.save_dir, 'route_sample_count.npy')
        route_sample_count_array = np.array([route_sample_count.get(i, 0) for i in range(self.num_routes)])
        np.save(route_count_file, route_sample_count_array)
        print(f"  每行样本数量已保存: {route_count_file}")

        return route_importances

    def prepare_training_data_from_routes(self, selected_route_indices):
        """
        只使用选定路线的历史数据准备训练数据
        """
        print(f"  准备训练数据，只使用选定路线...")
        history_data = self.matrix_data[:, :self.history_end]

        # 只收集选定路线的训练样本
        train_samples = []

        for route_idx in selected_route_indices:
            for t in range(self.history_end):
                val = history_data[route_idx, t]
                if not np.isnan(val) and val != 0:
                    train_samples.append((route_idx, t, val))

        if len(train_samples) == 0:
            print(f"  错误: 没有可用的训练样本")
            return None, None, None

        # 转换为 tensor
        route_indices = torch.tensor([s[0] for s in train_samples], dtype=torch.long).to(device)
        time_indices = torch.tensor([s[1] for s in train_samples], dtype=torch.long).to(device)
        values = torch.tensor([s[2] for s in train_samples], dtype=torch.float32).to(device).unsqueeze(1)

        # 确保索引形状正确
        if route_indices.dim() == 0:
            route_indices = route_indices.view(-1)
        if time_indices.dim() == 0:
            time_indices = time_indices.view(-1)

        print(f"  训练样本数: {len(train_samples)}")
        print(f"  使用的路线数: {len(selected_route_indices)}")

        return route_indices, time_indices, values

    def train_model_with_selected_routes(self, selected_route_indices, epochs=None, verbose=True):
        """
        只使用选定路线的数据重新训练模型
        """
        if epochs is None:
            epochs = self.epochs_per_step

        print(f"  使用选定路线重新训练模型...")

        # 准备训练数据（只使用选定路线）
        route_indices, time_indices, values = self.prepare_training_data_from_routes(selected_route_indices)

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

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Samples: {len(values)}, Batch: {batch_size}")

        print(f"  重新训练完成！最终损失: {avg_loss:.6f}")

        return avg_loss

    def experiment_with_routes(self, route_indices, exp_name):
        """使用指定的路线集合进行预测实验"""
        print(f"\n【{exp_name}】")
        print(f"  使用路线数: {len(route_indices)}")

        target_time = self.history_end  # 预测时间点 2001

        # 重新训练模型，只使用选定路线的数据
        print(f"  重新训练模型（只使用选定路线）...")
        print(f"  加载阶段1模型并初始化...")
        print(f"  使用阶段2训练epoch数: {self.stage2_epochs}")
        
        self.model = self.create_model()  # 创建新的模型
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 加载阶段1的模型权重
        if not self.load_model('trained_model.pth'):
            print(f"  警告: 未能加载阶段1模型，使用随机初始化")

        # 使用选定路线的数据训练（使用阶段2的epoch数）
        final_loss = self.train_model_with_selected_routes(route_indices, epochs=self.stage2_epochs, verbose=True)

        if final_loss is None:
            print(f"  训练失败，跳过实验")
            return None

        # 预测所有链路（不是只预测选定的路线）
        print(f"  预测所有链路在时间点 {target_time} 的值...")
        all_route_indices = np.arange(self.num_routes)
        predictions = self.predict_routes(all_route_indices, target_time)

        # 获取所有链路的真实值
        ground_truth = self.matrix_data[:, target_time]

        # 评估（只考虑有效值）
        valid_mask = ~np.isnan(ground_truth) & (ground_truth != 0)

        if valid_mask.any():
            valid_predictions = predictions[valid_mask]
            valid_ground_truth = ground_truth[valid_mask]

            mae = np.mean(np.abs(valid_predictions - valid_ground_truth))
            mse = np.mean((valid_predictions - valid_ground_truth) ** 2)
            rmse = np.sqrt(mse)
            
            # 计算相对误差
            mape = np.mean(np.abs(valid_predictions - valid_ground_truth) / np.abs(valid_ground_truth))

            print(f"  预测完成!")
            print(f"  训练使用路线数: {len(route_indices)}")
            print(f"  预测所有链路数: {self.num_routes}")
            print(f"  预测MAE: {mae:.6f}")
            print(f"  预测MSE: {mse:.6f}")
            print(f"  预测RMSE: {rmse:.6f}")
            print(f"  预测MAPE: {mape:.6f}")
            print(f"  有效预测数: {valid_mask.sum()}/{len(valid_mask)}")

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,  # 相对误差
                'valid_count': valid_mask.sum(),
                'total_count': len(valid_mask),
                'training_routes': len(route_indices),
                'prediction_routes': self.num_routes,
                'predictions': predictions  # 保存所有预测值
            }
        else:
            print(f"  警告: 没有有效的真实值")
            return None

    def run_online_learning(self):
        """运行训练和实验"""
        print("="*80)
        print("CostCo 训练与行重要性实验")
        print("="*80)
        print(f"数据形状: {self.matrix_data.shape}")
        print(f"历史数据: [0, {self.history_end - 1}]")
        print(f"预测时间点: {self.history_end} (即 2001)")
        print(f"采样率: {self.sample_rate}")
        print(f"使用代表点: {self.use_representer}")
        print(f"代表点方法: {self.representer_method}")
        print("="*80 + "\n")

        self.set_seed(self.global_seed)

        # 阶段1: 训练模型
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

        # 绘制样本重要性分布
        if self.sample_importances is not None:
            self.plot_importance_distribution(self.sample_importances)

        # 阶段2: 计算行（路线）重要性
        route_importances = self.compute_route_importance()

        # 阶段3: 两个对比实验
        print("\n" + "="*80)
        print(f"【阶段3】对比实验：{self.route_selection_ratio*100:.0f}% 最重要的行 vs 随机 {self.route_selection_ratio*100:.0f}% 行")
        print("="*80)

        # 计算需要选择的行数
        num_routes = self.num_routes
        num_top_routes = int(num_routes * self.route_selection_ratio)
        print(f"\n总行数: {num_routes}")
        print(f"选择的行数: {num_top_routes} ({self.route_selection_ratio*100:.0f}%)")

        # 实验1: 使用 40% 最重要的行
        print("\n" + "-"*80)
        top_route_indices = np.argsort(route_importances)[-num_top_routes:]
        result_top = self.experiment_with_routes(top_route_indices, f"实验1：{self.route_selection_ratio*100:.0f}% 最重要的行")

        # 实验2: 随机选择指定比例的行
        print("\n" + "-"*80)
        np.random.seed(self.global_seed + 1)
        random_route_indices = np.random.choice(num_routes, num_top_routes, replace=False)
        result_random = self.experiment_with_routes(random_route_indices, f"实验2：随机 {self.route_selection_ratio*100:.0f}% 的行")

        # 汇总对比结果
        print("\n" + "="*80)
        print("【实验结果对比】")
        print("="*80)
        if result_top and result_random:
            selection_ratio_str = f"{self.route_selection_ratio*100:.0f}%"
            print(f"\n说明: 越小的 MAE/MSE/RMSE/MAPE 越好，第四列表示 {selection_ratio_str} 最重要的行相对于随机选择是提升了还是下降了")
            print("-"*80)

            metrics = ['mae', 'mse', 'rmse', 'mape', 'valid_count']
            metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', '有效预测数']

            for metric, name in zip(metrics, metric_names):
                val_top = result_top[metric]
                val_random = result_random[metric]

                if metric == 'valid_count':
                    # 有效预测数：越大越好
                    if val_random > 0:
                        if val_top > val_random:
                            improvement = "提升了"
                        elif val_top < val_random:
                            improvement = "下降了"
                        else:
                            improvement = "持平"
                else:
                    # MAE/MSE/RMSE/MAPE：越小越好
                    if val_random > 0:
                        if val_top < val_random:
                            improvement = "提升了"
                        elif val_top > val_random:
                            improvement = "下降了"
                        else:
                            improvement = "持平"
                    else:
                        improvement = "N/A"

                if metric in ['mae', 'mse', 'rmse', 'mape']:
                    print(f"{name:<15} {val_top:<20.6f} {val_random:<20.6f} {improvement:<15}")
                else:
                    print(f"{name:<15} {val_top:<20} {val_random:<20} {improvement:<15}")

            print(f"\n说明: 越小的 MAE/MSE/RMSE/MAPE 越好，因此改进百分比表示 {self.route_selection_ratio*100:.0f}% 最重要的行相对于随机选择提升了多少")

        # 保存实验结果
        experiment_results = {
            'top_routes': {
                'indices': top_route_indices,
                'importances': route_importances[top_route_indices],
                'metrics': result_top
            },
            'random_routes': {
                'indices': random_route_indices,
                'importances': route_importances[random_route_indices],
                'metrics': result_random
            }
        }

        results_file = os.path.join(self.save_dir, 'experiment_results.npy')
        np.save(results_file, experiment_results)
        print(f"\n实验结果已保存: {results_file}")

        # 单独保存实验1和实验2的预测结果到独立文件夹，文件名包含时间点后缀
        if result_top and 'predictions' in result_top:
            # 实验1预测结果保存到 predictions_exp1_top_routes 文件夹
            target_time_str = f"t{self.history_end}"
            top_predictions_file = os.path.join(self.predictions_dir_exp1, f'predictions_{target_time_str}.npy')
            np.save(top_predictions_file, result_top['predictions'])
            print(f"实验1（重要行）预测结果已保存: {top_predictions_file}")

        if result_random and 'predictions' in result_random:
            # 实验2预测结果保存到 predictions_exp2_random_routes 文件夹
            target_time_str = f"t{self.history_end}"
            random_predictions_file = os.path.join(self.predictions_dir_exp2, f'predictions_{target_time_str}.npy')
            np.save(random_predictions_file, result_random['predictions'])
            print(f"实验2（随机行）预测结果已保存: {random_predictions_file}")

        # 绘制行重要性分布和对比图
        self.plot_route_importance_comparison(route_importances, top_route_indices, random_route_indices)

        # 保存配置
        self.save_config()

        print("\n" + "="*80)
        print("实验完成!")
        print("="*80)
    
    def save_model(self, filename):
        """保存模型（包括优化器状态）"""
        model_path = os.path.join(self.save_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'nc': self.nc,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            }
        }
        torch.save(checkpoint, model_path)
        print(f"  模型已保存: {model_path}")

    def load_model(self, filename):
        """加载模型（包括优化器状态）"""
        model_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(model_path):
            print(f"  警告: 模型文件不存在: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  模型已加载: {model_path}")
        return True

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
            f.write(f"\n实验说明:\n")
            f.write(f"- 计算每个样本的重要性（基于梯度）\n")
            f.write(f"- 计算每行的平均重要性（该行所有样本重要性的均值）\n")
            f.write(f"- 对比实验1：使用 {self.route_selection_ratio*100:.0f}% 最重要的行预测时间点 2001\n")
            f.write(f"- 对比实验2：使用随机 {self.route_selection_ratio*100:.0f}% 的行预测时间点 2001\n")
            f.write(f"- 预测目标：时间点 2001 的填充值\n")

        print(f"\n配置和种子信息已保存: {config_file}")

    def plot_importance_distribution(self, sample_importances):
        """绘制样本重要性分布"""
        print("  正在绘制样本重要性分布图...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 直方图
        axes[0, 0].hist(sample_importances, bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Sample Importance Distribution (Histogram)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Importance', fontsize=10)
        axes[0, 0].set_ylabel('Sample Count', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 排序
        sorted_importances = np.sort(sample_importances)[::-1]
        axes[0, 1].plot(range(len(sorted_importances)), sorted_importances, 'b-', linewidth=2)
        axes[0, 1].set_title('Sample Importance Ranking', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Sample Index (Sorted by Importance)', fontsize=10)
        axes[0, 1].set_ylabel('Importance', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 累积
        axes[1, 0].plot(np.arange(1, len(sample_importances) + 1),
                     np.cumsum(np.sort(sample_importances)[::-1]), 'b-', linewidth=2)
        axes[1, 0].set_title('Cumulative Importance Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Sample Count', fontsize=10)
        axes[1, 0].set_ylabel('Cumulative Importance', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 箱线图
        axes[1, 1].boxplot(sample_importances, patch_artist=None)
        axes[1, 1].set_title('Sample Importance Boxplot', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Importance', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_file = os.path.join(self.save_dir, 'sample_importance_distribution.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Sample importance distribution plot saved: {plot_file}")
        plt.close()

    def plot_route_importance_comparison(self, route_importances, top_route_indices, random_route_indices):
        """绘制行重要性对比图"""
        print("  正在绘制行重要性对比图...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        selection_ratio_str = f"{self.route_selection_ratio*100:.0f}%"

        # 1. 所有行的重要性分布
        axes[0, 0].hist(route_importances, bins=50, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('All Routes Importance Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Route Importance', fontsize=10)
        axes[0, 0].set_ylabel('Route Count', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 两种选择方式的对比
        sorted_indices = np.argsort(route_importances)
        sorted_importances = route_importances[sorted_indices]

        # 标记 Top 40% 的行
        top_mask = np.isin(sorted_indices, top_route_indices)
        random_mask = np.isin(sorted_indices, random_route_indices)

        colors = ['gray'] * len(sorted_importances)
        for i, (is_top, is_random) in enumerate(zip(top_mask, random_mask)):
            if is_top:
                colors[i] = 'red'
            elif is_random:
                colors[i] = 'green'

        axes[0, 1].scatter(range(len(sorted_importances)), sorted_importances,
                          c=colors, alpha=0.6, s=20)
        axes[0, 1].set_title(f'Route Importance Ranking (Red=Top {selection_ratio_str}, Green=Random {selection_ratio_str})', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Route Index (Sorted by Importance)', fontsize=10)
        axes[0, 1].set_ylabel('Route Importance', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Top 40% 行的重要性
        axes[1, 0].hist(route_importances[top_route_indices], bins=30, color='red', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title(f'Top {selection_ratio_str} Routes Importance Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Route Importance', fontsize=10)
        axes[1, 0].set_ylabel('Route Count', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 随机 40% 行的重要性
        axes[1, 1].hist(route_importances[random_route_indices], bins=30, color='green', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title(f'Random {selection_ratio_str} Routes Importance Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Route Importance', fontsize=10)
        axes[1, 1].set_ylabel('Route Count', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_file = os.path.join(self.save_dir, 'route_importance_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Route importance comparison plot saved: {plot_file}")
        plt.close()


def main():
    """主函数"""
    matrix_file = './output/Geant_23_23_3000_matrix_col_time_normalized.npy'
    
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
        'stage2_epochs': 20,  # 阶段2（实验阶段）训练的epoch数
        'loss_type': 'mae',
        
        # 代表点配置
        'use_representer': True,  # 使用代表点
        'representer_method': 'gradient',  # 梯度法
        'route_selection_ratio': 0.4,  # 选择路线的比例

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

    print("\n实验流程:")
    selection_ratio = config.get('route_selection_ratio', 0.4) * 100
    print("  1. 使用历史数据 [0, 2000) 训练 CostCo 模型")
    print("  2. 计算每个训练样本的重要性（基于梯度）")
    print("  3. 计算每行（路线）的平均重要性")
    print(f"  4. 实验1：使用 {selection_ratio:.0f}% 最重要的行重新训练模型，预测所有链路在时间点 2001 的值")
    print(f"  5. 实验2：使用随机 {selection_ratio:.0f}% 的行重新训练模型，预测所有链路在时间点 2001 的值")
    print("  6. 对比两个实验的预测效果（MAE, MSE, RMSE, MAPE）")

    learner = OnlineCostCoLearner(matrix_data, config)
    learner.run_online_learning()

    print("\n程序执行完成!")
    print("提示：")
    print("  - sample_importances.npy: 每个样本的重要性")
    print("  - route_importances.npy: 每行（路线）的平均重要性")
    print("  - route_sample_count.npy: 每行的训练样本数量")
    print("  - sample_importance_distribution.png: 样本重要性分布（4个子图）")
    print("  - route_importance_comparison.png: 行重要性对比（4个子图）")
    print("  - experiment_results.npy: 实验详细结果")
    print("  - predictions_exp1_top_routes/: 实验1（重要行）预测结果文件夹")
    print("    - predictions_t2001.npy: 实验1在时间点2001的所有预测值")
    print("  - predictions_exp2_random_routes/: 实验2（随机行）预测结果文件夹")
    print("    - predictions_t2001.npy: 实验2在时间点2001的所有预测值")


if __name__ == '__main__':
    main()
