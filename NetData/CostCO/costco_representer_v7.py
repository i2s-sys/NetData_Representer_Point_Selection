"""
CostCo 矩阵补全模型 - 这个版本 确定行相似性 + 混合路线选择策略
训练模型 -> 计算样本重要性 -> 计算行重要性 -> 混合路线选择（重要+不相似随机） vs 完全随机进行对比实验
说明：每个实验都会使用选定的路线重新训练模型，然后预测所有链路在指定时间点的值
支持批量执行：可循环执行多次，每次预测不同的时间点
混合策略：选择最重要的8%路线 + 从剩余92%路线中选择2%不相似路线（与重要路线相似度<0）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
from netsimilarity_utils import compute_similarity_mat_1d, get_k_nearest, compute_neighbor_count
matplotlib.use('Agg')

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


# ==================== 代表点重要性计算（优化版：批量计算独立梯度）====================
def compute_sample_importance_gradient(model, route_idx, time_idx, criterion, values):
    """
    计算每个样本的重要性（基于梯度）

    优化方案：批量前向传播，为每个样本计算独立梯度
    - 优点：保持100%准确性，同时显著提速
    - 方法：
        1. 批量前向传播（一次计算所有样本的predictions）
        2. 使用torch.autograd.grad为每个样本单独计算梯度
        3. 梯度完全独立，不累加

    Args:
        model: 模型
        route_idx: 路线索引
        time_idx: 时间索引
        criterion: 损失函数
        values: 目标值
    """
    batch_size = route_idx.size(0)

    # 保存原始参数
    original_route_embed = model.route_embeddings.weight.data.clone()
    original_time_embed = model.time_embeddings.weight.data.clone()

    sample_importances = []
    sample_grads = []

    model.eval()
    batch_size_per_iteration = min(64, batch_size)

    for i in range(0, batch_size, batch_size_per_iteration):
        end_idx = min(i + batch_size_per_iteration, batch_size)
        batch_routes = route_idx[i:end_idx]
        batch_times = time_idx[i:end_idx]
        batch_values = values[i:end_idx]

        with torch.enable_grad():
            # 批量前向传播（一次性计算所有样本的预测）
            predictions = model(batch_routes, batch_times)

            # 为每个样本计算独立梯度
            for j in range(len(batch_routes)):
                # 计算第j个样本的loss（scalar）
                loss_j = criterion(predictions[j:j+1], batch_values[j:j+1]).sum()

                # 使用autograd.grad计算该样本对嵌入权重的独立梯度
                # 这会返回一个梯度列表，每个对应一个需要求导的参数
                grads = torch.autograd.grad(
                    outputs=loss_j,
                    inputs=[model.route_embeddings.weight, model.time_embeddings.weight],
                    retain_graph=(j < len(batch_routes) - 1)  # 最后一个样本不需要保留
                )

                route_grad_j = grads[0][batch_routes[j], :].clone()  # 只取该样本对应的梯度
                time_grad_j = grads[1][batch_times[j], :].clone()   # 只取该样本对应的梯度

                # 计算重要性（基于梯度范数）
                route_imp = torch.norm(route_grad_j).item()
                time_imp = torch.norm(time_grad_j).item()
                imp = route_imp + time_imp
                sample_importances.append(imp)

                # 拼接梯度向量用于相似性计算
                grad_vector = torch.cat([route_grad_j, time_grad_j], dim=0).cpu().numpy()
                sample_grads.append(grad_vector)

    # 恢复原始参数
    model.route_embeddings.weight.data.copy_(original_route_embed)
    model.time_embeddings.weight.data.copy_(original_time_embed)

    sample_importances = np.array(sample_importances)
    sample_grads = np.array(sample_grads)

    # 归一化
    sample_importances = sample_importances / (sample_importances.mean() + 1e-8)

    grad_info = {
        'route_grad_norm': np.linalg.norm(sample_grads[:, :64]) if len(sample_grads) > 0 else 0,
        'time_grad_norm': np.linalg.norm(sample_grads[:, 64:]) if len(sample_grads) > 0 else 0,
        'mean_importance': sample_importances.mean(),
        'std_importance': sample_importances.std(),
        'min_importance': sample_importances.min(),
        'max_importance': sample_importances.max()
    }

    return sample_importances, sample_grads, grad_info

# ==================== 样本相似性计算（优化版：GPU加速+向量化）====================
def compute_similarity_from_grads(sample_grads, k_neighbors=5, similarity_threshold=0.8, max_samples=50000):
    """
    基于已计算的梯度向量计算样本相似性矩阵
    优化：使用GPU加速和向量化计算，限制最大样本数以避免内存溢出

    Args:
        sample_grads: 样本梯度向量 [N, 2*embedding_dim]
        k_neighbors: k近邻数量
        similarity_threshold: 相似性阈值（用于硬邻居计数）
        max_samples: 最大样本数限制（避免内存溢出）

    Returns:
        similarity_mat: 相似性矩阵 [N, N] 或 [N_sampled, N_sampled]
        k_nearest_indices: 每个样本的k个最近邻索引 [N, k] 或 [N_sampled, k]
        neighbor_counts: 每个样本的邻居数量 [N] 或 [N_sampled]
        sampled_indices: 采样的样本索引（如果进行了采样）
    """
    num_samples = len(sample_grads)

    # 如果样本数量超过限制，进行随机采样
    if num_samples > max_samples:
        print(f"  样本数量 {num_samples} 超过限制 {max_samples}，进行随机采样...")
        np.random.seed(42)
        sampled_indices = np.random.choice(num_samples, size=max_samples, replace=False)
        sampled_indices.sort()
        sample_grads = sample_grads[sampled_indices]
        num_samples = max_samples
        print(f"  采样后样本数量: {num_samples}")
    else:
        sampled_indices = None

    print("  计算样本相似性矩阵（优化版，GPU加速）...")
    print(f"  样本数量: {num_samples}")
    print(f"  梯度维度: {sample_grads.shape[1]}")

    # 转换为torch tensor并移到GPU
    grads_tensor = torch.tensor(sample_grads, dtype=torch.float32).to(device)

    # 在GPU上计算余弦相似度（向量化）
    # 归一化
    grad_norms = torch.norm(grads_tensor, p=2, dim=1, keepdim=True)
    grad_norms = torch.clamp(grad_norms, min=1e-8)  # 避免除以零
    grad_norms = grads_tensor / grad_norms

    # 计算相似性矩阵（批量矩阵乘法）
    similarity_mat = torch.mm(grad_norms, grad_norms.T)

    # 转换回numpy
    similarity_mat = similarity_mat.cpu().numpy()

    print(f"  相似性矩阵形状: {similarity_mat.shape}")
    print(f"  相似性范围: [{similarity_mat.min():.6f}, {similarity_mat.max():.6f}]")

    # 使用向量化方法找到k个最相似的邻居（避免循环）
    # 对每行找到最大的k个值的索引
    similarity_mat_tensor = torch.from_numpy(similarity_mat).to(device)
    k_nearest_indices_tensor = torch.topk(similarity_mat_tensor, k=k_neighbors, dim=1).indices
    k_nearest_indices = k_nearest_indices_tensor.cpu().numpy()

    # 如果进行了采样，需要将索引映射回原始样本
    if sampled_indices is not None:
        k_nearest_indices = sampled_indices[k_nearest_indices]

    # 计算每个样本的邻居数量（基于阈值，向量化）
    neighbor_counts = (similarity_mat > similarity_threshold).sum(axis=1)

    print(f"  k近邻索引形状: {k_nearest_indices.shape}")
    print(f"  邻居数量范围: [{neighbor_counts.min():.0f}, {neighbor_counts.max():.0f}]")
    print(f"  相似性计算完成!")

    return similarity_mat, k_nearest_indices, neighbor_counts, sampled_indices


# ==================== 路线相似性计算（基于样本相似性 + 时间对齐）====================
def compute_route_similarity_from_sample_similarity(
    similarity_mat,
    route_indices,
    time_indices,
    num_routes,
    method='time_aligned',
    sample_importances=None,
    k_neighbors=None
):
    """
    从样本相似性矩阵计算路线相似性矩阵（基于时间对齐的新方案）

    Args:
        similarity_mat: 样本相似性矩阵 [N_samples, N_samples]
        route_indices: 每个样本对应的路线索引 [N_samples]
        time_indices: 每个样本对应的时间索引 [N_samples]
        num_routes: 总路线数
        method: 聚合方法，可选：
            - 'time_aligned': 基于时间对齐的聚合（新方案，默认）
            - 'mean': 平均相似性（旧方案）
            - 'max': 最大相似性
            - 'min': 最小相似性
            - 'weighted_mean': 加权平均（基于样本重要性）
        sample_importances: 样本重要性 [N_samples]，用于加权平均
        k_neighbors: k近邻数量（保留参数以兼容旧接口）

    Returns:
        route_similarity_mat: 路线相似性矩阵 [num_routes, num_routes]
        route_neighbor_counts: 每条路线的相似路线数量 [num_routes]
    """
    print("\\n【路线相似性计算】")
    print(f"  样本相似性矩阵形状: {similarity_mat.shape}")
    print(f"  路线索引形状: {route_indices.shape}")
    print(f"  总路线数: {num_routes}")
    print(f"  聚合方法: {method}")

    num_samples = len(route_indices)

    if method == 'time_aligned':
        print("  使用基于时间对齐的新方案...")

        # 初始化路线相似性矩阵和计数矩阵
        route_similarity_sum = np.zeros((num_routes, num_routes))
        route_similarity_count = np.zeros((num_routes, num_routes))

        # 对每条路线遍历其所有样本点
        print(f"  遍历路线进行时间对齐聚合...")
        for route_r in range(num_routes):
            if route_r % 50 == 0:
                print(f"    进度: {route_r}/{num_routes} ({route_r/num_routes*100:.1f}%)")

            # 找到路线r的所有样本索引
            samples_of_route_r = np.where(route_indices == route_r)[0]

            if len(samples_of_route_r) == 0:
                continue

            # 遍历路线r的每个样本点
            for sample_i in samples_of_route_r:
                time_i = time_indices[sample_i]

                # 找到同一时间点的所有样本
                samples_at_time_i = np.where(time_indices == time_i)[0]

                # 对于同一时间点的每个样本j，累加相似度到对应路线
                for sample_j in samples_at_time_i:
                    route_k = route_indices[sample_j]

                    # 只计算与其他路线的相似度（不包括自己）
                    if route_k != route_r:
                        # 累加相似度
                        route_similarity_sum[route_r, route_k] += similarity_mat[sample_i, sample_j]
                        # 计数+1（公共时间点）
                        route_similarity_count[route_r, route_k] += 1

        # 计算最终路线相似性矩阵
        # 避免除以零（理论上不应该发生，因为529条路线前100个时间点都有数据）
        route_similarity_count[route_similarity_count == 0] = 1
        route_similarity_mat = route_similarity_sum / route_similarity_count

        # 设置自己与自己的相似性为-1（防止自匹配）
        for route_r in range(num_routes):
            route_similarity_mat[route_r, route_r] = -1.0

        # 对称化（因为相似性应该是对称的）
        # 注意：对角线仍然是-1
        for i in range(num_routes):
            for j in range(i + 1, num_routes):
                avg_sim = (route_similarity_mat[i, j] + route_similarity_mat[j, i]) / 2
                route_similarity_mat[i, j] = avg_sim
                route_similarity_mat[j, i] = avg_sim

    else:
        # 保留旧方案以兼容
        print("  使用旧方案（遍历所有样本对）...")

        # 初始化路线相似性矩阵和计数矩阵
        route_similarity_sum = np.zeros((num_routes, num_routes))
        route_similarity_count = np.zeros((num_routes, num_routes))

        # 遍历所有样本对，按路线聚合
        print(f"  遍历样本对进行聚合...")
        for i in range(num_samples):
            if i % 5000 == 0:
                print(f"    进度: {i}/{num_samples} ({i/num_samples*100:.1f}%)")

            route_i = route_indices[i]
            time_i = time_indices[i]

            for j in range(num_samples):
                route_j = route_indices[j]
                time_j = time_indices[j]

                # 只聚合不同路线的样本对（避免同一路线的自相似性影响）
                if route_i != route_j:
                    similarity = similarity_mat[i, j]

                    if method == 'mean':
                        # 平均聚合
                        route_similarity_sum[route_i, route_j] += similarity
                        route_similarity_count[route_i, route_j] += 1

                    elif method == 'max':
                        # 最大聚合
                        route_similarity_sum[route_i, route_j] = max(
                            route_similarity_sum[route_i, route_j],
                            similarity
                        )
                        route_similarity_count[route_i, route_j] = 1  # 标记已更新

                    elif method == 'min':
                        # 最小聚合
                        if route_similarity_count[route_i, route_j] == 0:
                            route_similarity_sum[route_i, route_j] = similarity
                        else:
                            route_similarity_sum[route_i, route_j] = min(
                                route_similarity_sum[route_i, route_j],
                                similarity
                            )
                        route_similarity_count[route_i, route_j] = 1

                    elif method == 'weighted_mean':
                        # 加权平均聚合（基于样本重要性）
                        if sample_importances is not None:
                            weight = sample_importances[i] * sample_importances[j]
                            route_similarity_sum[route_i, route_j] += similarity * weight
                            route_similarity_count[route_i, route_j] += weight
                        else:
                            # 如果没有样本重要性，退化为普通平均
                            route_similarity_sum[route_i, route_j] += similarity
                            route_similarity_count[route_i, route_j] += 1

        # 计算最终路线相似性矩阵
        if method in ['mean', 'weighted_mean']:
            # 避免除以零
            route_similarity_count[route_similarity_count == 0] = 1
            route_similarity_mat = route_similarity_sum / route_similarity_count
        else:
            # max/min方法直接使用sum矩阵
            route_similarity_mat = route_similarity_sum

        # 对称化（因为相似性是对称的）
        route_similarity_mat = (route_similarity_mat + route_similarity_mat.T) / 2

        # 设置自己与自己的相似性为-1（防止自匹配）
        for route_r in range(num_routes):
            route_similarity_mat[route_r, route_r] = -1.0

    # 计算每条路线的相似路线数量（基于阈值）
    similarity_threshold = 0.7  # 可调整的阈值
    route_neighbor_counts = np.sum(route_similarity_mat > similarity_threshold, axis=1)  # 不需要减1，因为对角线是-1

    print(f"  路线相似性矩阵形状: {route_similarity_mat.shape}")
    print(f"  路线相似性范围: [{route_similarity_mat.min():.6f}, {route_similarity_mat.max():.6f}]")
    print(f"  相似路线数量范围: [{route_neighbor_counts.min():.0f}, {route_neighbor_counts.max():.0f}]")
    print(f"  平均相似路线数: {route_neighbor_counts.mean():.2f}")

    return route_similarity_mat, route_neighbor_counts


# ==================== 样本索引查询辅助函数====================
def get_sample_info_from_index(sample_idx, route_indices, time_indices, values=None):
    """
    从样本索引获取样本的详细信息

    Args:
        sample_idx: 样本在相似性矩阵中的索引
        route_indices: 所有样本的路线索引数组
        time_indices: 所有样本的时间索引数组
        values: 所有样本的值数组（可选）

    Returns:
        样本信息字典
    """
    sample_info = {
        'sample_idx': sample_idx,
        'route_idx': int(route_indices[sample_idx]),
        'time_idx': int(time_indices[sample_idx])
    }

    if values is not None:
        sample_info['value'] = float(values[sample_idx])

    return sample_info


# ==================== 随机采样方法（支持代表点）====================
def random_sampling_with_representer(matrix_data, seed_num, sample_rate=0.8, min_train_samples=100, use_representer=False):
    """
    随机采样（支持代表点）
    优化：先找到所有有效位置，再从有效位置中采样
    """
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)

    num_routes, num_time = matrix_data.shape

    # 先找到所有有效位置（非0非NaN）
    valid_positions = []
    for r in range(num_routes):
        for t in range(num_time):
            val = matrix_data[r, t]
            if not np.isnan(val) and val != 0:
                valid_positions.append((r, t, val))

    num_valid = len(valid_positions)

    if num_valid == 0:
        print("  警告: 没有找到有效样本（非0非NaN）")
        return []

    # 从有效位置中采样
    num_train = int(num_valid * sample_rate)
    num_train = max(num_train, min_train_samples)

    # 如果有效样本数少于需要的训练样本数，使用所有有效样本
    if num_train > num_valid:
        num_train = num_valid
        print(f"  警告: 有效样本数({num_valid})少于期望的训练样本数，使用所有有效样本")

    # 随机选择样本
    selected_indices = np.random.choice(num_valid, num_train, replace=False)
    train_samples = [valid_positions[i] for i in selected_indices]

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
        self.lr = config.get('lr', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-7)
        self.epochs_per_step = config.get('epochs_per_step', 50)
        self.history_start = config.get('history_start', 0)
        self.history_end = config.get('history_end', 2000)
        self.history_end = min(self.history_end, self.num_time)
        self.save_dir = config.get('save_dir', './online_results_representer')
        self.top_level_dir = config.get('top_level_dir', './online_results_representer')  # 顶层目录，用于统一保存预测结果和模型
        self.sample_rate = config.get('sample_rate', 0.8)
        self.loss_type = config.get('loss_type', 'mae')
        self.global_seed = config.get('global_seed', 42)
        self.min_train_samples = config.get('min_train_samples', 100)
        self.use_representer = config.get('use_representer', True)
        self.representer_method = config.get('representer_method', 'gradient')
        self.route_selection_ratio = config.get('route_selection_ratio', 0.1)  # 总路线选择比例
        self.top_route_ratio = config.get('top_route_ratio', 0.09)  # 最重要路线的比例（8%）
        self.random_route_ratio = config.get('random_route_ratio', 0.01)  # 从随机路线中选择的比例（2%）
        self.dissimilar_threshold = config.get('dissimilar_threshold', 0.0)  # 不相似阈值
        self.exp3_random_ratio = config.get('exp3_random_ratio', 0.01)  # 实验3：从重要路线中选择的随机路线比例（2%）
        self.exp4_importance_ratio = config.get('exp4_importance_ratio', 0.1)  # 实验4：重要路线比例（10%）
        self.stage2_epochs = config.get('stage2_epochs', 100)  # 阶段2（实验阶段）训练的epoch数
        # 收敛判断参数
        self.patience = config.get('patience', 10)  # 验证损失连续N轮没有下降就认为收敛
        self.val_split = config.get('val_split', 0.2)  # 验证集比例

        # 相似性计算参数
        self.use_similarity = config.get('use_similarity', True)  # 是否计算样本相似性
        self.k_neighbors = config.get('k_neighbors', 5)  # k近邻数量
        self.similarity_threshold = config.get('similarity_threshold', 0.8)  # 相似性阈值（用于硬邻居计数）
        self.max_similarity_samples = config.get('max_similarity_samples', 50000)  # 最大样本数限制（避免内存溢出）
        self.top_similar_routes = config.get('top_similar_routes', 3)  # 分析最重要路线相似性时，检查每条路线的前top个相似路线

        os.makedirs(self.save_dir, exist_ok=True)

        # 创建顶层预测结果保存目录（统一管理所有时间点的预测结果，基于顶层目录）
        self.top_predictions_dir = os.path.join(self.top_level_dir, 'predictions_top_routes')
        self.random_predictions_dir = os.path.join(self.top_level_dir, 'predictions_random_routes')
        os.makedirs(self.top_predictions_dir, exist_ok=True)
        os.makedirs(self.random_predictions_dir, exist_ok=True)

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

        # 初始化优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

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
        训练模型 - 使用代表点重要性，支持基于验证损失的早停收敛判断
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

        # 划分训练集和验证集
        num_samples = len(values)
        num_val = int(num_samples * self.val_split)
        num_train = num_samples - num_val

        # 随机打乱数据
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        # 创建训练集和验证集
        train_route_indices = route_indices[train_indices]
        train_time_indices = time_indices[train_indices]
        train_values = values[train_indices]

        val_route_indices = route_indices[val_indices]
        val_time_indices = time_indices[val_indices]
        val_values = values[val_indices]

        batch_size = min(128, num_train)
        if batch_size < 10:
            batch_size = num_train

        train_dataset = torch.utils.data.TensorDataset(train_route_indices, train_time_indices, train_values)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if self.loss_type == 'mae':
            criterion = CustomLoss('mae')
        elif self.loss_type == 'mse':
            criterion = CustomLoss('mse')
        elif self.loss_type == 'mae_mse':
            criterion = CustomLoss('mae_mse')

        # 更新训练样本索引为训练集部分（用于计算行重要性）
        self.train_route_indices = train_route_indices.cpu().numpy()
        self.train_time_indices = train_time_indices.cpu().numpy()

        # 保存训练数据供后续使用
        self.train_route_indices_tensor = train_route_indices
        self.train_time_indices_tensor = train_time_indices
        self.train_values_tensor = train_values

        epoch_importances = []

        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        train_loss_history = []
        val_loss_history = []

        self.model.train()

        print(f"  训练样本数: {num_train}, 验证样本数: {num_val}")
        print(f"  收敛判断: 连续 {self.patience} 轮验证损失没有下降则停止")
        print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")

        for epoch in range(epochs):
            # 训练阶段
            total_loss = 0
            self.model.train()

            for batch_routes, batch_times, batch_values in train_dataloader:
                self.optimizer.zero_grad()

                predictions = self.model(batch_routes, batch_times)
                loss = criterion(predictions, batch_values)

                total_loss += loss.sum().item()

                loss.mean().backward()
                self.optimizer.step()

            avg_train_loss = total_loss / num_train

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_route_indices, val_time_indices)
                val_loss_batch = criterion(val_predictions, val_values)
                avg_val_loss = val_loss_batch.sum().item() / num_val

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)

            # 判断是否收敛（基于验证损失的下降）
            is_best = False
            if avg_val_loss < best_val_loss:
                # 只有验证损失下降时才更新最佳损失并重置计数器
                best_val_loss = avg_val_loss
                patience_counter = 0
                is_best = True
            else:
                # 验证损失没有下降，计数器+1
                patience_counter += 1

            # 计算损失变化幅度（用于日志显示）
            val_loss_change = abs(avg_val_loss - best_val_loss)

            if verbose:
                print(f"  Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                      f"Best Val: {best_val_loss:.6f}, Change: {val_loss_change:.6f}, Patience: {patience_counter}/{self.patience}")

            # 早停判断
            if patience_counter >= self.patience:
                print(f"\n  模型收敛！验证损失连续 {self.patience} 轮没有下降")
                print(f"  最终训练损失: {avg_train_loss:.6f}")
                print(f"  最终验证损失: {avg_val_loss:.6f}")
                print(f"  最佳验证损失: {best_val_loss:.6f}")
                print(f"  实际训练轮数: {epoch+1}/{epochs}")
                break

        # 模型收敛后，计算样本重要性和相似度矩阵
        if self.use_representer:
            print("\n【模型收敛后计算样本重要性和相似度】")

            # 使用收敛后的模型计算样本重要性
            print("  计算样本重要性...")
            sample_importances, sample_grads, grad_info = compute_sample_importance_gradient(
                self.model,
                self.train_route_indices_tensor,
                self.train_time_indices_tensor,
                criterion,
                self.train_values_tensor
            )
            self.sample_importances = sample_importances

            print(f"  样本重要性统计:")
            print(f"    平均: {grad_info['mean_importance']:.6f}")
            print(f"    标准差: {grad_info['std_importance']:.6f}")
            print(f"    最小: {grad_info['min_importance']:.6f}")
            print(f"    最大: {grad_info['max_importance']:.6f}")
            print(f"    链路梯度范数: {grad_info['route_grad_norm']:.6f}")
            print(f"    时间梯度范数: {grad_info['time_grad_norm']:.6f}")

            # 计算样本相似性矩阵（基于收敛后的梯度）
            if self.use_similarity:
                print("\n  计算样本相似性矩阵...")
                similarity_mat, k_nearest_indices, neighbor_counts, sampled_indices = compute_similarity_from_grads(
                    sample_grads,
                    k_neighbors=self.k_neighbors,
                    similarity_threshold=self.similarity_threshold,
                    max_samples=self.max_similarity_samples
                )

                # 保存相似性信息
                self.similarity_mat = similarity_mat
                self.k_nearest_indices = k_nearest_indices
                self.neighbor_counts = neighbor_counts

                print(f"  样本相似性计算完成!")
                print(f"    相似性矩阵形状: {similarity_mat.shape}")
                print(f"    k近邻索引形状: {k_nearest_indices.shape}")
                print(f"    邻居数量形状: {neighbor_counts.shape}")
                print(f"    相似性范围: [{similarity_mat.min():.6f}, {similarity_mat.max():.6f}]")
                print(f"    邻居数量范围: [{neighbor_counts.min():.0f}, {neighbor_counts.max():.0f}]")

                # 计算路线相似性矩阵（基于样本相似性 + 时间对齐）
                print("\n  计算路线相似性矩阵...")
                self.route_similarity_mat, self.route_neighbor_counts = compute_route_similarity_from_sample_similarity(
                    similarity_mat,
                    self.train_route_indices,
                    self.train_time_indices,
                    self.num_routes,
                    method='time_aligned',  # 新方案：基于时间对齐的聚合
                    sample_importances=None,  # 新方案不使用样本重要性
                    k_neighbors=self.k_neighbors  # 保留参数以兼容
                )

                # 保存路线相似性信息
                route_similarity_file = os.path.join(self.save_dir, 'route_similarity_mat.npy')
                np.save(route_similarity_file, self.route_similarity_mat)
                print(f"  路线相似性矩阵已保存: {route_similarity_file}")

                route_neighbor_counts_file = os.path.join(self.save_dir, 'route_neighbor_counts.npy')
                np.save(route_neighbor_counts_file, self.route_neighbor_counts)
                print(f"  路线邻居数量已保存: {route_neighbor_counts_file}")
            else:
                self.similarity_mat = None
                self.k_nearest_indices = None
                self.neighbor_counts = None
                self.route_similarity_mat = None
                self.route_neighbor_counts = None

        # 恢复训练模式
        self.model.train()

        return avg_train_loss, train_loss_history, val_loss_history

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

    def analyze_similarity_among_top_routes(self, top_route_indices, route_importances, top_k=10):
        """
        分析最重要路线之间的相似性

        对于每条最重要路线，找出其top_k相似路线，检查这些相似路线是否也在最重要路线集合中

        Args:
            top_route_indices: 最重要路线的索引列表
            route_importances: 所有路线的重要性
            top_k: 每条路线检查前k个相似路线
        """
        if self.route_similarity_mat is None:
            print("  路线相似性矩阵未计算，跳过相似性分析")
            return

        print("\n【分析最重要路线之间的相似性】")
        print(f"  最重要路线数量: {len(top_route_indices)}")
        print(f"  检查每条路线的前{top_k}个相似路线")

        # 将最重要路线转换为集合，方便快速查找
        top_route_set = set(top_route_indices)

        # 用于存储结果
        similar_pairs = []
        route_similar_count = {}

        # 对每条最重要路线进行分析
        for i, route_a in enumerate(top_route_indices):
            # 获取路线a的相似性向量
            similarity_vector = self.route_similarity_mat[route_a, :]

            # 找到top_k最相似的路线（不包括自己，因为自相似性是-1）
            # 使用argsort找到相似度最高的k个索引
            top_k_indices = np.argsort(similarity_vector)[-top_k:]

            # 从后往前遍历（相似度从高到低）
            for route_b in reversed(top_k_indices):
                if route_b == route_a:
                    continue  # 跳过自己

                # 检查路线b是否也在最重要路线集合中
                if route_b in top_route_set:
                    # 记录这一对路线
                    similarity_score = similarity_vector[route_b]
                    similar_pairs.append({
                        'route_a': route_a,
                        'route_b': route_b,
                        'similarity': similarity_score,
                        'importance_a': route_importances[route_a],
                        'importance_b': route_importances[route_b]
                    })

                    # 统计每条路线与其他重要路线的相似数量
                    if route_a not in route_similar_count:
                        route_similar_count[route_a] = 0
                    route_similar_count[route_a] += 1

        # 输出所有符合条件的路线对
        print(f"\n  共找到 {len(similar_pairs)} 对重要路线之间的相似关系:")
        print("-" * 100)

        if len(similar_pairs) > 0:
            for i, pair in enumerate(similar_pairs, 1):
                print(f"  {i}. 最重要的 {pair['route_a']} 路线与 {pair['route_b']} 路线相似，两条路线都属于最重要的行")
                print(f"     相似度: {pair['similarity']:.6f}")
                print(f"     路线{pair['route_a']}重要性: {pair['importance_a']:.6f}")
                print(f"     路线{pair['route_b']}重要性: {pair['importance_b']:.6f}")
                print()
        else:
            print("  没有找到重要路线之间的相似关系（重要路线之间彼此不相似）")

        print("-" * 100)

        # 统计信息
        print(f"\n  统计信息:")
        if route_similar_count:
            avg_similar_count = sum(route_similar_count.values()) / len(route_similar_count)
            print(f"    平均每条重要路线与其他重要路线相似的数量: {avg_similar_count:.2f}")
            print(f"    与其他重要路线相似数量最多的路线:")
            max_similar = max(route_similar_count.items(), key=lambda x: x[1])
            print(f"      路线 {max_similar[0]}: 与 {max_similar[1]} 条其他重要路线相似")

        # 保存结果到文件
        result_file = os.path.join(self.save_dir, 'top_route_similarity_analysis.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"最重要路线之间的相似性分析\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"最重要路线数量: {len(top_route_indices)}\n")
            f.write(f"检查每条路线的前{top_k}个相似路线\n\n")
            f.write(f"共找到 {len(similar_pairs)} 对重要路线之间的相似关系:\n")
            f.write("-" * 100 + "\n\n")

            for i, pair in enumerate(similar_pairs, 1):
                f.write(f"{i}. 最重要的 {pair['route_a']} 路线与 {pair['route_b']} 路线相似，两条路线都属于最重要的行\n")
                f.write(f"   相似度: {pair['similarity']:.6f}\n")
                f.write(f"   路线{pair['route_a']}重要性: {pair['importance_a']:.6f}\n")
                f.write(f"   路线{pair['route_b']}重要性: {pair['importance_b']:.6f}\n\n")

            if route_similar_count:
                avg_similar_count = sum(route_similar_count.values()) / len(route_similar_count)
                f.write(f"\n统计信息:\n")
                f.write(f"  平均每条重要路线与其他重要路线相似的数量: {avg_similar_count:.2f}\n")
                max_similar = max(route_similar_count.items(), key=lambda x: x[1])
                f.write(f"  与其他重要路线相似数量最多的路线: {max_similar[0]} (与{max_similar[1]}条其他重要路线相似)\n")

        print(f"\n  分析结果已保存到: {result_file}")

        return similar_pairs

    def select_mixed_routes(self, route_importances, from_top_routes=False):
        """
        混合路线选择策略：
        1. 选择最重要的 top_route_ratio 路线（如8%）
        2. 根据参数选择额外的路线：
           - from_top_routes=False: 从剩余路线中选择 random_route_ratio 路线（如2%），要求与重要路线不相似
           - from_top_routes=True: 从重要路线中选择 random_route_ratio 路线（如2%），纯随机

        Args:
            route_importances: 所有路线的重要性
            from_top_routes: 是否从重要路线中选择额外路线
                         - False: 从剩余路线中选择（实验1）
                         - True: 从重要路线中选择（实验3）

        Returns:
            selected_routes: 选中的路线索引
            top_routes: 最重要路线索引
            other_routes: 其他路线索引（不相似随机路线或重要路线中的随机路线）
        """
        print("\n【混合路线选择策略】")
        print(f"  最重要路线比例: {self.top_route_ratio*100:.0f}%")
        if from_top_routes:
            print(f"  随机路线比例: {self.random_route_ratio*100:.0f}%（从重要路线中选择）")
        else:
            print(f"  不相似随机路线比例: {self.random_route_ratio*100:.0f}%（从剩余路线中选择）")
            print(f"  不相似阈值: {self.dissimilar_threshold}")

        num_routes = len(route_importances)

        # 计算需要选择的路线数量
        num_top_routes = int(num_routes * self.top_route_ratio)
        num_other_routes = int(num_routes * self.random_route_ratio)
        num_total_routes = num_top_routes + num_other_routes

        print(f"  最重要路线数量: {num_top_routes}")
        print(f"  其他路线数量: {num_other_routes}")
        print(f"  总选择路线数量: {num_total_routes}")

        # 1. 选择最重要的路线
        top_routes = np.argsort(route_importances)[-num_top_routes:]
        print(f"  已选择最重要路线: {top_routes}")

        # 2. 根据参数选择额外路线
        if from_top_routes:
            # 实验3：从重要路线中选择随机路线
            candidate_pool = top_routes
            print(f"  候选路线池大小: {len(candidate_pool)} (从重要路线中选择)")
            np.random.seed(self.global_seed)
            other_routes = np.random.choice(candidate_pool, num_other_routes, replace=False)
            print(f"  已从重要路线中随机选择: {other_routes}")
        else:
            # 实验1：从剩余路线中选择不相似路线
            all_routes = np.arange(num_routes)
            candidate_pool = np.setdiff1d(all_routes, top_routes)
            remaining_percentage = (1 - self.top_route_ratio) * 100
            print(f"  剩余路线池大小: {len(candidate_pool)} (占总路线的{remaining_percentage:.0f}%)")

            # 3. 从候选池中筛选不相似的路线
            if self.route_similarity_mat is None:
                print("  警告: 路线相似性矩阵未计算，使用随机选择")
                other_routes = np.random.choice(candidate_pool, num_other_routes, replace=False)
            else:
                dissimilar_candidates = []
                for candidate in candidate_pool:
                    # 检查该候选路线与所有重要路线的相似度
                    similarities = self.route_similarity_mat[candidate, top_routes]
                    # 只保留与所有重要路线相似度都小于阈值的路线
                    if np.all(similarities < self.dissimilar_threshold):
                        dissimilar_candidates.append(candidate)

                print(f"  满足不相似条件的路线数量: {len(dissimilar_candidates)}")

                # 如果不相似路线不足，降低阈值或使用全部候选路线
                if len(dissimilar_candidates) < num_other_routes:
                    print(f"  警告: 不相似路线不足 ({len(dissimilar_candidates)} < {num_other_routes})")
                    # 尝试降低阈值
                    original_threshold = self.dissimilar_threshold
                    while len(dissimilar_candidates) < num_other_routes and self.dissimilar_threshold < 1.0:
                        self.dissimilar_threshold += 0.1
                        dissimilar_candidates = []
                        for candidate in candidate_pool:
                            similarities = self.route_similarity_mat[candidate, top_routes]
                            if np.all(similarities < self.dissimilar_threshold):
                                dissimilar_candidates.append(candidate)
                        print(f"  降低阈值到 {self.dissimilar_threshold:.1f}, 满足条件的路线数量: {len(dissimilar_candidates)}")

                    # 如果仍然不足，使用全部候选路线
                    if len(dissimilar_candidates) < num_other_routes:
                        print(f"  警告: 即使降低阈值也不足，使用全部候选路线")
                        dissimilar_candidates = list(candidate_pool)

                    # 恢复原始阈值
                    self.dissimilar_threshold = original_threshold

                # 从不相似路线中随机选择指定数量
                np.random.seed(self.global_seed)
                if len(dissimilar_candidates) >= num_other_routes:
                    other_routes = np.random.choice(dissimilar_candidates, num_other_routes, replace=False)
                else:
                    # 候选路线仍然不足，使用所有候选路线
                    other_routes = np.array(dissimilar_candidates)
                    print(f"  实际选择不相似路线: {len(other_routes)}")

            print(f"  已选择不相似随机路线: {other_routes}")

        # 4. 合并路线
        selected_routes = np.concatenate([top_routes, other_routes])

        # 验证不相似路线确实与重要路线相似度 < 阈值（仅当 from_top_routes=False 时）
        if not from_top_routes and self.route_similarity_mat is not None:
            print(f"\n  验证不相似路线:")
            for other_route in other_routes:
                similarities = self.route_similarity_mat[other_route, top_routes]
                max_similarity = similarities.max()
                if max_similarity >= self.dissimilar_threshold:
                    print(f"    警告: 路线 {other_route} 与重要路线的最大相似度为 {max_similarity:.6f} >= {self.dissimilar_threshold}")

        return selected_routes, top_routes, other_routes

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
        只使用选定路线的数据重新训练模型，支持基于验证损失的早停收敛判断
        """
        if epochs is None:
            epochs = self.epochs_per_step

        print(f"  使用选定路线重新训练模型...")

        # 准备训练数据（只使用选定路线）
        route_indices, time_indices, values = self.prepare_training_data_from_routes(selected_route_indices)

        if values is None:
            return None, None, None

        # 划分训练集和验证集
        num_samples = len(values)
        num_val = int(num_samples * self.val_split)
        num_train = num_samples - num_val

        # 随机打乱数据
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        # 创建训练集和验证集
        train_route_indices = route_indices[train_indices]
        train_time_indices = time_indices[train_indices]
        train_values = values[train_indices]

        val_route_indices = route_indices[val_indices]
        val_time_indices = time_indices[val_indices]
        val_values = values[val_indices]

        batch_size = min(128, num_train)
        if batch_size < 10:
            batch_size = num_train

        train_dataset = torch.utils.data.TensorDataset(train_route_indices, train_time_indices, train_values)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if self.loss_type == 'mae':
            criterion = CustomLoss('mae')
        elif self.loss_type == 'mse':
            criterion = CustomLoss('mse')
        elif self.loss_type == 'mae_mse':
            criterion = CustomLoss('mae_mse')

        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        train_loss_history = []
        val_loss_history = []

        self.model.train()

        print(f"  训练样本数: {num_train}, 验证样本数: {num_val}")
        print(f"  收敛判断: 连续 {self.patience} 轮验证损失没有下降则停止")
        print(f"  固定学习率: {self.optimizer.param_groups[0]['lr']:.2e}")

        for epoch in range(epochs):
            # 训练阶段
            total_loss = 0
            self.model.train()

            for batch_routes, batch_times, batch_values in train_dataloader:
                predictions = self.model(batch_routes, batch_times)
                loss = criterion(predictions, batch_values)

                total_loss += loss.sum().item()

                loss.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_train_loss = total_loss / num_train

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_route_indices, val_time_indices)
                val_loss_batch = criterion(val_predictions, val_values)
                avg_val_loss = val_loss_batch.sum().item() / num_val

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)

            # 判断是否收敛（基于验证损失的下降，与阶段1保持一致）
            is_best = False
            if avg_val_loss < best_val_loss:
                # 只有验证损失下降时才更新最佳损失并重置计数器
                best_val_loss = avg_val_loss
                patience_counter = 0
                is_best = True
            else:
                # 验证损失没有下降，计数器+1
                patience_counter += 1

            # 计算损失变化幅度（用于日志显示）
            val_loss_change = abs(avg_val_loss - best_val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                      f"Best Val: {best_val_loss:.6f}, Change: {val_loss_change:.6f}, Patience: {patience_counter}/{self.patience}")

            # 早停判断
            if patience_counter >= self.patience:
                print(f"\n  模型收敛！验证损失连续 {self.patience} 轮没有下降")
                print(f"  最终训练损失: {avg_train_loss:.6f}")
                print(f"  最终验证损失: {avg_val_loss:.6f}")
                print(f"  最佳验证损失: {best_val_loss:.6f}")
                print(f"  实际训练轮数: {epoch+1}/{epochs}")
                break

        # 恢复训练模式
        self.model.train()

        print(f"  重新训练完成！最终损失: {avg_train_loss:.6f}, 最佳验证损失: {best_val_loss:.6f}")

        return avg_train_loss

    def experiment_with_routes(self, route_indices, exp_name, use_stage1_init=True):
        """使用指定的路线集合进行预测实验

        Args:
            route_indices: 选择的路线索引
            exp_name: 实验名称
            use_stage1_init: 是否使用阶段1训练好的模型初始化（True则加载，False则随机初始化）
        """
        print(f"\n【{exp_name}】")
        print(f"  使用路线数: {len(route_indices)}")

        target_time = self.history_end  # 预测时间点

        # 重新训练模型，只使用选定路线的数据
        print(f"  重新训练模型（只使用选定路线）...")

        if use_stage1_init:
            print(f"  使用阶段1训练好的模型初始化...")
            print(f"  使用阶段2训练epoch数（早停机制）: {self.stage2_epochs}")
        else:
            print(f"  使用随机初始化（不加载阶段1模型）...")
            print(f"  使用阶段2训练epoch数（早停机制）: {self.stage2_epochs}")

        self.model = self.create_model()  # 创建新的模型
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 根据参数决定是否加载阶段1的模型权重
        if use_stage1_init:
            # 实验1：加载阶段1的模型权重
            if not self.load_model('trained_model.pth'):
                print(f"  警告: 未能加载阶段1模型，使用随机初始化")
        else:
            # 实验2：不加载阶段1模型，使用随机初始化
            print(f"  模型已随机初始化，未加载阶段1权重")

        # 使用选定路线的数据训练（使用阶段2的epoch数，由早停机制决定实际轮数）
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
        print(f"预测时间点: {self.history_end}")
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
        final_avg_loss, train_loss_history, val_loss_history = self.train_model_with_representer(epochs=self.epochs_per_step, verbose=True)
        print(f"\n训练完成！")
        print(f"最终平均训练损失: {final_avg_loss:.6f}")
        print(f"训练历史: Train Loss从{train_loss_history[0]:.6f}到{train_loss_history[-1]:.6f}")
        print(f"验证历史: Val Loss从{val_loss_history[0]:.6f}到{val_loss_history[-1]:.6f}")

        self.save_model('trained_model.pth')

        # 保存样本重要性
        if self.sample_importances is not None:
            importance_file = os.path.join(self.save_dir, 'sample_importances.npy')
            np.save(importance_file, self.sample_importances)
            print(f"样本重要性已保存: {importance_file}")

        # 绘制样本重要性分布（已禁用）
        # if self.sample_importances is not None:
        #     self.plot_importance_distribution(self.sample_importances)

        # 绘制路线相似性矩阵可视化（已禁用）
        # if self.route_similarity_mat is not None:
        #     self.plot_route_similarity_matrix(self.route_similarity_mat, self.route_neighbor_counts)

        # 阶段2: 计算行（路线）重要性
        route_importances = self.compute_route_importance()

        # 阶段3: 对比实验
        print("\n" + "="*80)
        print("【阶段3】对比实验")
        print("="*80)

        # 计算需要选择的行数
        num_routes = self.num_routes
        print(f"\n总行数: {num_routes}")

        # 实验1: 使用混合路线选择策略（使用阶段1模型初始化）
        print("\n" + "-"*80)
        print("【实验1】")
        selected_routes_exp1, top_routes_exp1, dissimilar_routes_exp1 = self.select_mixed_routes(route_importances, from_top_routes=False)
        result_exp1 = self.experiment_with_routes(selected_routes_exp1, f"实验1：{self.top_route_ratio*100:.0f}% 重要 + {self.random_route_ratio*100:.0f}% 不相似随机", use_stage1_init=True)

        # 实验2: 随机选择指定比例的行（随机初始化，不使用阶段1模型）
        print("\n" + "-"*80)
        print("【实验2】")
        np.random.seed(self.global_seed + 1)
        num_routes_total = int(num_routes * self.route_selection_ratio)
        random_route_indices = np.random.choice(num_routes, num_routes_total, replace=False)
        result_exp2 = self.experiment_with_routes(random_route_indices, f"实验2：完全随机 {self.route_selection_ratio*100:.0f}% 的行", use_stage1_init=False)

        # 实验3: 从重要路线中选择随机路线（使用阶段1模型初始化）
        print("\n" + "-"*80)
        print("【实验3】")
        selected_routes_exp3, top_routes_exp3, random_routes_exp3 = self.select_mixed_routes(route_importances, from_top_routes=True)
        result_exp3 = self.experiment_with_routes(selected_routes_exp3, f"实验3：{self.top_route_ratio*100:.0f}% 重要 + {self.random_route_ratio*100:.0f}% 随机", use_stage1_init=True)

        # 实验4: 选择最重要的10%路线（使用阶段1模型初始化）
        print("\n" + "-"*80)
        print("【实验4】")
        num_routes_exp4 = int(num_routes * self.exp4_importance_ratio)
        top_routes_exp4 = np.argsort(route_importances)[-num_routes_exp4:]
        result_exp4 = self.experiment_with_routes(top_routes_exp4, f"实验4：{self.exp4_importance_ratio*100:.0f}% 最重要", use_stage1_init=True)

        # 汇总对比结果
        print("\n" + "="*80)
        print("【实验结果对比】")
        print("="*80)
        if result_exp1 and result_exp2 and result_exp3 and result_exp4:
            print(f"\n说明: 越小的 MAE/MSE/RMSE/MAPE 越好")
            print("-"*80)

            metrics = ['mae', 'mse', 'rmse', 'mape']
            metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE']

            print(f"{'实验名称':<15} {'实验1':<20} {'实验2':<20} {'实验3':<20} {'实验4':<20}")
            print("-"*100)

            for metric, name in zip(metrics, metric_names):
                val_exp1 = result_exp1[metric]
                val_exp2 = result_exp2[metric]
                val_exp3 = result_exp3[metric]
                val_exp4 = result_exp4[metric]

                print(f"{name:<15} {val_exp1:<20.6f} {val_exp2:<20.6f} {val_exp3:<20.6f} {val_exp4:<20.6f}")

        # 保存实验结果
        experiment_results = {
            'exp1': {
                'indices': selected_routes_exp1,
                'importances': route_importances[selected_routes_exp1],
                'metrics': result_exp1
            },
            'exp2': {
                'indices': random_route_indices,
                'importances': route_importances[random_route_indices],
                'metrics': result_exp2
            },
            'exp3': {
                'indices': selected_routes_exp3,
                'importances': route_importances[selected_routes_exp3],
                'metrics': result_exp3
            },
            'exp4': {
                'indices': top_routes_exp4,
                'importances': route_importances[top_routes_exp4],
                'metrics': result_exp4
            }
        }

        results_file = os.path.join(self.save_dir, 'experiment_results.npy')
        np.save(results_file, experiment_results)
        print(f"\n实验结果已保存: {results_file}")

        # 单独保存实验1和实验2的预测结果到顶层目录
        if result_exp1 and 'predictions' in result_exp1:
            # 实验1预测结果保存到顶层 predictions_top_routes 文件夹
            target_time_str = f"t{self.history_end}"
            top_predictions_file = os.path.join(self.top_predictions_dir, f'predictions_{target_time_str}.npy')
            np.save(top_predictions_file, result_exp1['predictions'])
            print(f"实验1（混合策略）预测结果已保存: {top_predictions_file}")

        if result_exp2 and 'predictions' in result_exp2:
            # 实验2预测结果保存到顶层 predictions_random_routes 文件夹
            target_time_str = f"t{self.history_end}"
            random_predictions_file = os.path.join(self.random_predictions_dir, f'predictions_{target_time_str}.npy')
            np.save(random_predictions_file, result_exp2['predictions'])
            print(f"实验2（完全随机）预测结果已保存: {random_predictions_file}")

        if result_exp3 and 'predictions' in result_exp3:
            # 实验3预测结果保存到顶层 predictions_top_routes 文件夹
            target_time_str = f"t{self.history_end}"
            exp3_predictions_file = os.path.join(self.top_predictions_dir, f'predictions_{target_time_str}.npy')
            np.save(exp3_predictions_file, result_exp3['predictions'])
            print(f"实验3（重要+随机）预测结果已保存: {exp3_predictions_file}")

        if result_exp4 and 'predictions' in result_exp4:
            # 实验4预测结果保存到顶层 predictions_top_routes 文件夹
            target_time_str = f"t{self.history_end}"
            exp4_predictions_file = os.path.join(self.top_predictions_dir, f'predictions_{target_time_str}.npy')
            np.save(exp4_predictions_file, result_exp4['predictions'])
            print(f"实验4（最重要）预测结果已保存: {exp4_predictions_file}")

        # 绘制行重要性分布和对比图（已禁用）
        # self.plot_route_importance_comparison(route_importances, selected_routes, random_route_indices)

        # 保存配置
        self.save_config()

        print("\n" + "="*80)
        print("实验完成!")
        print("="*80)

    def save_model(self, filename):
        """保存模型到顶层目录（覆盖之前的模型）"""
        # 模型保存到顶层目录，每次覆盖
        model_path = os.path.join(self.top_level_dir, filename)
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
        print(f"  模型已保存（顶层目录）: {model_path}")

    def load_model(self, filename):
        """从顶层目录加载模型（包括优化器状态）"""
        model_path = os.path.join(self.top_level_dir, filename)
        if not os.path.exists(model_path):
            print(f"  警告: 模型文件不存在: {model_path}")
            return False

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  模型已加载（顶层目录）: {model_path}")
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
            f.write(f"- 对比实验1：使用 {self.top_route_ratio*100:.0f}% 最重要路线 + {self.random_route_ratio*100:.0f}% 不相似随机路线（从剩余路线中选择）预测时间点 {self.history_end}\n")
            f.write(f"- 对比实验2：使用完全随机 {self.route_selection_ratio*100:.0f}% 的路线预测时间点 {self.history_end}\n")
            f.write(f"- 对比实验3：使用 {self.top_route_ratio*100:.0f}% 最重要路线 + {self.random_route_ratio*100:.0f}% 随机路线（从重要路线中选择）预测时间点 {self.history_end}\n")
            f.write(f"- 对比实验4：使用 {self.exp4_importance_ratio*100:.0f}% 最重要路线预测时间点 {self.history_end}\n")
            f.write(f"- 预测目标：时间点 {self.history_end} 的填充值\n")

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

    def plot_route_similarity_matrix(self, route_similarity_mat, route_neighbor_counts):
        """绘制路线相似性矩阵可视化"""
        print("  正在绘制路线相似性矩阵可视化...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 热力图（完整矩阵）
        im1 = axes[0, 0].imshow(route_similarity_mat, cmap='hot', aspect='auto')
        axes[0, 0].set_title('Route Similarity Matrix (Heatmap)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Route Index', fontsize=10)
        axes[0, 0].set_ylabel('Route Index', fontsize=10)
        plt.colorbar(im1, ax=axes[0, 0], label='Similarity')

        # 2. 直方图（相似性分布）
        # 只取上三角部分（排除对角线）
        upper_triangle = route_similarity_mat[np.triu_indices_from(route_similarity_mat, k=1)]
        axes[0, 1].hist(upper_triangle, bins=50, color='orange', edgecolor='black')
        axes[0, 1].set_title('Route Similarity Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Similarity', fontsize=10)
        axes[0, 1].set_ylabel('Count', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=0.7, color='r', linestyle='--', label='Threshold=0.7')
        axes[0, 1].legend()

        # 3. 每条路线的相似路线数量
        axes[1, 0].bar(range(len(route_neighbor_counts)), route_neighbor_counts, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Number of Similar Routes per Route', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Route Index', fontsize=10)
        axes[1, 0].set_ylabel('Number of Similar Routes', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. 平均相似性（每条路线）
        avg_similarity_per_route = np.mean(route_similarity_mat, axis=1)
        axes[1, 1].plot(range(len(avg_similarity_per_route)), avg_similarity_per_route, 'b-', linewidth=1.5)
        axes[1, 1].set_title('Average Similarity per Route', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Route Index', fontsize=10)
        axes[1, 1].set_ylabel('Average Similarity', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_file = os.path.join(self.save_dir, 'route_similarity_visualization.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Route similarity visualization plot saved: {plot_file}")
        plt.close()

        # 统计信息
        print(f"\\n路线相似性统计:")
        print(f"  总路线数: {len(route_neighbor_counts)}")
        print(f"  平均相似性: {np.mean(upper_triangle):.6f}")
        print(f"  相似性标准差: {np.std(upper_triangle):.6f}")
        print(f"  最小相似性: {np.min(upper_triangle):.6f}")
        print(f"  最大相似性: {np.max(upper_triangle):.6f}")
        print(f"  平均相似路线数: {np.mean(route_neighbor_counts):.2f}")
        print(f"  相似路线数中位数: {np.median(route_neighbor_counts):.2f}")
        print(f"  孤立路线数（相似路线数=0）: {np.sum(route_neighbor_counts == 0)}")
        print(f"  高度相似路线数（相似路线数>10）: {np.sum(route_neighbor_counts > 10)}")

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


def plot_summary_results(all_results_summary, save_dir):
    """绘制所有时间点的汇总结果"""
    print("  正在绘制汇总结果图...")

    target_times = all_results_summary['target_times']
    top_mae = all_results_summary['top_routes_metrics']['mae']
    random_mae = all_results_summary['random_routes_metrics']['mae']

    if not target_times:
        print("  警告: 没有可用的结果数据")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ['mae', 'mse', 'rmse', 'mape']
    metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE']

    # 前4个子图：每个指标的对比
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        top_values = all_results_summary['top_routes_metrics'][metric]
        random_values = all_results_summary['random_routes_metrics'][metric]

        if top_values and random_values:
            ax.plot(target_times, top_values, 'b-', label=f'Important Routes ({name})', linewidth=1.5, alpha=0.7)
            ax.plot(target_times, random_values, 'r--', label=f'Random Routes ({name})', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time Point', fontsize=10)
            ax.set_ylabel(name, fontsize=10)
            ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    # 第5个子图：改进百分比
    ax = axes[0, 2]
    improvements = []
    for i in range(len(target_times)):
        top_val = all_results_summary['top_routes_metrics']['mae'][i]
        random_val = all_results_summary['random_routes_metrics']['mae'][i]
        if random_val > 0:
            improvement = ((random_val - top_val) / random_val) * 100
            improvements.append(improvement)

    if improvements:
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax.bar(target_times[:len(improvements)], improvements, color=colors, alpha=0.6, width=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time Point', fontsize=10)
        ax.set_ylabel('Improvement (%)', fontsize=10)
        ax.set_title('MAE Improvement Percentage', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # 第6个子图：平均改进率
    ax = axes[1, 2]
    metrics_to_plot = ['mae', 'mse', 'rmse', 'mape']
    avg_improvements = []

    for metric in metrics_to_plot:
        top_vals = all_results_summary['top_routes_metrics'][metric]
        random_vals = all_results_summary['random_routes_metrics'][metric]
        if top_vals and random_vals:
            improvements = [((r - t) / r * 100) for t, r in zip(top_vals, random_vals) if r > 0]
            avg_improvements.append(np.mean(improvements) if improvements else 0)
        else:
            avg_improvements.append(0)

    colors_bar = ['green' if imp > 0 else 'red' for imp in avg_improvements]
    bars = ax.bar(metric_names, avg_improvements, color=colors_bar, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Average Improvement (%)', fontsize=10)
    ax.set_title('Average Improvement by Metric', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, value in zip(bars, avg_improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(save_dir, 'summary_all_timepoints.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  汇总结果图已保存: {plot_file}")
    plt.close()


def main():
    """主函数 - 执行1000次循环，预测时间点2000-2999"""
    # PMU_59_28_4965_matrix_col_time_normalized 1 Geant_23_23_3000_matrix_col_time_normalized
    # Seattle_99_99_688_matrix_col_time_normalized 1 Abilene_12_12_3000_matrix_col_time_normalized
    # PlanetLab_489_489_17_matrix_col_time_normalized 需注意时间点比较少 应该测试10-15
    # PlanetLab_matrix_col_time_50_50_normalized 需注意时间点比较少 应该测试10-15
    # Seattle_matrix_col_time_50_50_normalized 时间点多可以测试50-70
    # Seattle_matrix_col_time_30_30_normalized
    # PlanetLab_matrix_col_time_30_30_normalized
    matrix_file = './output/Seattle_99_99_688_matrix_col_time_normalized.npy' # ./output/Abilene_12_12_3000_matrix_col_time_normalized.npy' # Geant_23_23_3000_matrix_col_time_normalized.npy'

    print(f"加载数据文件: {matrix_file}")
    if not os.path.exists(matrix_file):
        print(f"错误: 数据文件不存在: {matrix_file}")
        return

    matrix_data = np.load(matrix_file)
    print(f"数据加载成功，形状: {matrix_data.shape}")
    print(f"数据类型: {matrix_data.dtype}")
    print(f"数据范围: [{np.nanmin(matrix_data):.2f}, {np.nanmax(matrix_data):.2f}]")

    # 预测时间点范围：2980-2999（共20个时间点）
    target_time_range = range(5, 9)
    total_iterations = len(target_time_range)

    # 创建汇总结果保存目录
    summary_save_dir = './online_results_representer_summary'
    os.makedirs(summary_save_dir, exist_ok=True)

    # 用于汇总所有时间点的结果
    all_results_summary = {
        'target_times': [],
        'exp1_metrics': {'mae': [], 'mse': [], 'rmse': [], 'mape': []},
        'exp2_metrics': {'mae': [], 'mse': [], 'rmse': [], 'mape': []},
        'exp3_metrics': {'mae': [], 'mse': [], 'rmse': [], 'mape': []},
        'exp4_metrics': {'mae': [], 'mse': [], 'rmse': [], 'mape': []}
    }

    # 记录总开始时间
    overall_start_time = time.time()

    print("="*80)
    print(f"开始执行 {total_iterations} 次完整实验")
    print(f"预测时间点范围: {target_time_range.start} - {target_time_range.stop-1}")
    print("="*80)
    print(f"\n存储优化说明:")
    print(f"  - 预测结果统一保存到顶层: ./online_results_representer/predictions_top_routes 和 predictions_random_routes")
    print(f"  - 模型文件统一保存到顶层（每次覆盖）: ./online_results_representer/trained_model.pth")
    print(f"  - 每个时间点的详细结果保存到独立子目录")
    print("="*80)

    for idx, target_time in enumerate(target_time_range):
        print("\n" + "="*80)
        print(f"进度: [{idx+1}/{total_iterations}] 预测时间点: {target_time}")
        print(f"训练数据范围: [0, {target_time})")
        print("="*80)

        # 记录当前时间点开始时间
        timepoint_start_time = time.time()

        config = {
            # 模型参数
            'embedding_dim': 64,
            'nc': 128,

            # 训练参数
            'lr': 1e-3,
            'weight_decay': 1e-7,
            'epochs_per_step': 100,  # 优化：减少最大epoch数，由早停机制决定实际训练轮数
            'stage2_epochs': 100,  # 优化：阶段2训练epoch数，也由早停机制决定
            'loss_type': 'mae',

            # 收敛判断参数
            'patience': 10,  # 验证损失连续N轮没有下降则停止
            'val_split': 0.2,  # 验证集比例

            # 相似性计算参数
            'use_similarity': True,  # 是否计算样本相似性（禁用以避免内存爆炸：True会计算N×N矩阵，需要300GB+内存）
            'k_neighbors': 5,  # k近邻数量
            'similarity_threshold': 0.8,  # 相似性阈值（用于硬邻居计数）
            'max_similarity_samples': 100000,  # 最大样本数限制（避免内存溢出）
            'top_similar_routes': 1,  # 分析最重要路线相似性时，检查每条路线的前top个相似路线

            # 代表点配置
            'use_representer': True,  # 使用代表点
            'representer_method': 'gradient',  # 梯度法
            'route_selection_ratio': 0.1,  # 总路线选择比例
            'top_route_ratio': 0.08,  # 最重要路线的比例（8%）  尝试9 1 和8 2
            'random_route_ratio': 0.02,  # 从随机路线中选择的比例（2%）
            'dissimilar_threshold': 0.0,  # 不相似阈值（相似度小于此值才可选）
            'exp3_random_ratio': 0.02,  # 实验3：从重要路线中选择的随机路线比例（2%）
            'exp4_importance_ratio': 0.1,  # 实验4：重要路线比例（10%）

            # 采样配置
            'sample_rate': 1,
            'global_seed': 42,
            'min_train_samples': 100,

            # 数据划分 - 动态更新
            'history_start': 0,
            'history_end': target_time,  # 关键：训练到目标时间点之前

            # 保存 - 为每个时间点创建独立目录
            'save_dir': f'./online_results_representer/t{target_time}',
            'top_level_dir': './online_results_representer'  # 顶层目录，统一保存预测结果和模型
        }

        # 创建并运行学习者
        learner = OnlineCostCoLearner(matrix_data, config)

        # 修改experiment_with_routes方法以支持返回结果
        print("\n配置参数:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行实验
        learner.run_online_learning()

        # 获取该时间点的实验结果
        experiment_file = os.path.join(config['save_dir'], 'experiment_results.npy')
        if os.path.exists(experiment_file):
            experiment_data = np.load(experiment_file, allow_pickle=True).item()

            # 汇总结果
            all_results_summary['target_times'].append(target_time)

            # 记录实验1的指标
            if experiment_data['exp1']['metrics'] is not None:
                all_results_summary['exp1_metrics']['mae'].append(experiment_data['exp1']['metrics']['mae'])
                all_results_summary['exp1_metrics']['mse'].append(experiment_data['exp1']['metrics']['mse'])
                all_results_summary['exp1_metrics']['rmse'].append(experiment_data['exp1']['metrics']['rmse'])
                all_results_summary['exp1_metrics']['mape'].append(experiment_data['exp1']['metrics']['mape'])

            # 记录实验2的指标
            if experiment_data['exp2']['metrics'] is not None:
                all_results_summary['exp2_metrics']['mae'].append(experiment_data['exp2']['metrics']['mae'])
                all_results_summary['exp2_metrics']['mse'].append(experiment_data['exp2']['metrics']['mse'])
                all_results_summary['exp2_metrics']['rmse'].append(experiment_data['exp2']['metrics']['rmse'])
                all_results_summary['exp2_metrics']['mape'].append(experiment_data['exp2']['metrics']['mape'])

            # 记录实验3的指标
            if experiment_data['exp3']['metrics'] is not None:
                all_results_summary['exp3_metrics']['mae'].append(experiment_data['exp3']['metrics']['mae'])
                all_results_summary['exp3_metrics']['mse'].append(experiment_data['exp3']['metrics']['mse'])
                all_results_summary['exp3_metrics']['rmse'].append(experiment_data['exp3']['metrics']['rmse'])
                all_results_summary['exp3_metrics']['mape'].append(experiment_data['exp3']['metrics']['mape'])

            # 记录实验4的指标
            if experiment_data['exp4']['metrics'] is not None:
                all_results_summary['exp4_metrics']['mae'].append(experiment_data['exp4']['metrics']['mae'])
                all_results_summary['exp4_metrics']['mse'].append(experiment_data['exp4']['metrics']['mse'])
                all_results_summary['exp4_metrics']['rmse'].append(experiment_data['exp4']['metrics']['rmse'])
                all_results_summary['exp4_metrics']['mape'].append(experiment_data['exp4']['metrics']['mape'])

            print(f"\n时间点 {target_time} 的实验结果已汇总")

    # 保存所有时间点的汇总结果
    summary_file = os.path.join(summary_save_dir, 'all_timepoints_summary.npy')
    np.save(summary_file, all_results_summary)
    print(f"\n所有时间点的汇总结果已保存: {summary_file}")

    # 绘制汇总结果图（已禁用）
    # plot_summary_results(all_results_summary, summary_save_dir)

    print("\n" + "="*80)
    print(f"全部 {total_iterations} 次实验完成!")
    print("="*80)
    print("\n汇总统计:")
    print(f"  - 预测时间点范围: {target_time_range.start} - {target_time_range.stop-1}")
    print(f"  - 有效实验次数: {len(all_results_summary['target_times'])}")
    print(f"\n平均指标对比:")
    metrics = ['mae', 'mse', 'rmse', 'mape']
    metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE']

    print(f"{'指标名称':<15} {'实验1':<20} {'实验2':<20} {'实验3':<20} {'实验4':<20}")
    print("-"*80)

    for metric, name in zip(metrics, metric_names):
        exp1_avg = np.mean(all_results_summary['exp1_metrics'][metric]) if all_results_summary['exp1_metrics'][metric] else 0
        exp2_avg = np.mean(all_results_summary['exp2_metrics'][metric]) if all_results_summary['exp2_metrics'][metric] else 0
        exp3_avg = np.mean(all_results_summary['exp3_metrics'][metric]) if all_results_summary['exp3_metrics'][metric] else 0
        exp4_avg = np.mean(all_results_summary['exp4_metrics'][metric]) if all_results_summary['exp4_metrics'][metric] else 0

        print(f"{name:<15} {exp1_avg:<20.6f} {exp2_avg:<20.6f} {exp3_avg:<20.6f} {exp4_avg:<20.6f}")
    print(f"\n所有结果已保存到: {summary_save_dir}")
    print(f"  - all_timepoints_summary.npy: 汇总数据")
    print(f"  - summary_*.png: 汇总可视化图表")
    print(f"\n预测结果（所有时间点）:")
    print(f"  - ./online_results_representer/predictions_top_routes/: 重要路线的所有预测结果（t2000.npy, t2001.npy, ...）")
    print(f"  - ./online_results_representer/predictions_random_routes/: 随机路线的所有预测结果（t2000.npy, t2001.npy, ...）")
    print(f"\n共享文件（顶层，每次覆盖）:")
    print(f"  - ./online_results_representer/trained_model.pth: 最新训练的模型")
    print(f"\n每个时间点的详细结果（不包含模型和预测）:")
    print(f"  - ./online_results_representer/tXXXX/: 样本重要性、路线重要性、可视化图等")


if __name__ == '__main__':
    main()
