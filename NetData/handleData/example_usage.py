"""
示例：如何使用转换后的矩阵进行训练
演示与现有代码（Amain.py, config.py）的集成
"""

import numpy as np
import torch
import torch.nn as nn
from handleData.tensor_converter import load_converted_matrix, TensorConverter


def example_1_load_and_inspect():
    """示例1: 加载并检查转换后的矩阵"""
    print("="*70)
    print("示例1: 加载并检查转换后的矩阵")
    print("="*70 + "\n")
    
    # 加载不同模式的矩阵
    modes = ['flatten', 'user_time', 'item_time', 'time_user_item', 'avg']
    
    for mode in modes:
        try:
            matrix = load_converted_matrix('Abilene', mode=mode)
            print(f"模式 {mode:20s}: 形状 {str(matrix.shape):20s}, "
                  f"范围 [{matrix.min():.4f}, {matrix.max():.4f}], "
                  f"均值 {matrix.mean():.4f}")
        except FileNotFoundError:
            print(f"模式 {mode:20s}: 文件未找到，需要先转换")
    
    print()


def example_2_train_simple_model():
    """示例2: 使用转换后的矩阵训练简单模型"""
    print("="*70)
    print("示例2: 使用转换后的矩阵训练简单模型")
    print("="*70 + "\n")
    
    # 加载数据
    try:
        matrix = load_converted_matrix('Abilene', mode='flatten')
    except FileNotFoundError:
        print("矩阵文件未找到，先进行转换...")
        converter = TensorConverter('Abilene')
        matrix = converter.convert_to_matrix(mode='flatten', save=True)
    
    # 转换为 PyTorch 张量
    data = torch.from_numpy(matrix).float()
    
    # 归一化（参考 config.py 的归一化方式）
    max_val = data.max()
    data_normalized = data / max_val
    
    print(f"原始数据形状: {data.shape}")
    print(f"归一化后范围: [{data_normalized.min():.6f}, {data_normalized.max():.6f}]")
    
    # 分割数据集
    n_samples = data_normalized.shape[0]
    train_size = int(0.8 * n_samples)
    valid_size = int(0.1 * n_samples)
    
    train_data = data_normalized[:train_size]
    valid_data = data_normalized[train_size:train_size + valid_size]
    test_data = data_normalized[train_size + valid_size:]
    
    print(f"\n数据集分割:")
    print(f"  训练集: {train_data.shape}")
    print(f"  验证集: {valid_data.shape}")
    print(f"  测试集: {test_data.shape}")
    
    # 定义简单的时间序列预测模型
    class SimplePredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    # 使用前 T-1 个时间步预测最后一个时间步
    T = data_normalized.shape[1]
    model = SimplePredictor(T - 1)
    
    if torch.cuda.is_available():
        model = model.cuda()
        train_data = train_data.cuda()
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练
    print(f"\n开始训练...")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        
        # 前向传播
        X_train = train_data[:, :-1]  # 前 T-1 个时间步
        y_train = train_data[:, -1]    # 最后一个时间步
        
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                X_valid = valid_data[:, :-1]
                y_valid = valid_data[:, -1]
                pred_valid = model(X_valid)
                valid_loss = criterion(pred_valid, y_valid)
                
                print(f"Epoch [{epoch+1:3d}/{epochs}]  "
                      f"Train Loss: {loss.item():.6f}  "
                      f"Valid Loss: {valid_loss.item():.6f}")
    
    # 测试
    print(f"\n测试...")
    model.eval()
    with torch.no_grad():
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        pred_test = model(X_test)
        test_loss = criterion(pred_test, y_test)
        
        # 计算 NMAE
        error = torch.abs(pred_test - y_test)
        nmae = error.sum() / y_test.abs().sum()
        
        print(f"Test Loss: {test_loss.item():.6f}")
        print(f"Test NMAE: {nmae.item():.6f}")
    
    print()


def example_3_convert_new_dataset():
    """示例3: 转换新的数据集"""
    print("="*70)
    print("示例3: 转换新的数据集")
    print("="*70 + "\n")
    
    # 转换 Geant 数据集
    dataset_name = 'Geant'
    
    try:
        converter = TensorConverter(dataset_name)
        print(f"成功加载 {dataset_name} 数据集")
        print(f"原始张量形状: {converter.tensor.shape}")
        
        # 转换为矩阵
        matrix = converter.convert_to_matrix(mode='flatten', save=True)
        print(f"转换后矩阵形状: {matrix.shape}")
        
        # 也可以转换所有模式
        # converter.convert_all_modes()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请确保 {dataset_name}.npy 和 {dataset_name}.ini 文件存在于 data/ 目录")
    
    print()


def example_4_comparison_with_original():
    """示例4: 对比原始张量和转换后的矩阵"""
    print("="*70)
    print("示例4: 对比原始张量和转换后的矩阵")
    print("="*70 + "\n")
    
    # 加载原始张量
    import os
    tensor_path = './data/Abilene.npy'
    
    if os.path.exists(tensor_path):
        original_tensor = np.load(tensor_path)
        print(f"原始张量形状: {original_tensor.shape}")
        print(f"原始张量范围: [{original_tensor.min():.4f}, {original_tensor.max():.4f}]")
        print(f"原始张量均值: {original_tensor.mean():.4f}")
        print(f"原始张量非零值比例: {(original_tensor != 0).sum() / original_tensor.size:.2%}")
        print()
        
        # 转换后的矩阵
        converter = TensorConverter('Abilene')
        matrix = converter.convert_to_matrix(mode='flatten', save=False)
        
        print(f"转换后矩阵形状: {matrix.shape}")
        print(f"转换后矩阵范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
        print(f"转换后矩阵均值: {matrix.mean():.4f}")
        print(f"转换后矩阵非零值比例: {(matrix != 0).sum() / matrix.size:.2%}")
        print()
        
        # 验证数据一致性
        print("验证数据一致性:")
        print(f"  总元素数相等: {original_tensor.size == matrix.size}")
        print(f"  最大值相等: {np.isclose(original_tensor.max(), matrix.max())}")
        print(f"  最小值相等: {np.isclose(original_tensor.min(), matrix.min())}")
        print(f"  均值相等: {np.isclose(original_tensor.mean(), matrix.mean())}")
        print()
    else:
        print(f"原始张量文件不存在: {tensor_path}")
        print()


def example_5_different_modes_for_different_tasks():
    """示例5: 不同模式适合不同任务"""
    print("="*70)
    print("示例5: 不同模式适合不同任务")
    print("="*70 + "\n")
    
    tasks = {
        'flatten': {
            'description': '每个样本是一个用户-物品对的完整时间序列',
            'suitable_for': ['时间序列预测', '序列分类', '模式识别'],
            'model_types': ['LSTM', 'Transformer', 'RNN', 'GRU']
        },
        'user_time': {
            'description': '每个样本是一个用户在所有物品上的时间序列',
            'suitable_for': ['用户行为分析', '用户画像', '个性化推荐'],
            'model_types': ['FCN', 'Autoencoder', 'CNN']
        },
        'item_time': {
            'description': '每个样本是一个物品被所有用户使用的时间序列',
            'suitable_for': ['物品分析', '服务监控', '资源调度'],
            'model_types': ['FCN', 'Time Series Models', 'Anomaly Detection']
        },
        'time_user_item': {
            'description': '每个样本是一个时间点的所有用户-物品快照',
            'suitable_for': ['动态分析', '趋势预测', '时空建模'],
            'model_types': ['GCN', 'Spatial-Temporal Models', 'GNN']
        },
        'avg': {
            'description': '每个用户-物品对是一个标量（时间平均）',
            'suitable_for': ['静态推荐', '矩阵补全', '协同过滤'],
            'model_types': ['Matrix Factorization', 'Collaborative Filtering', 'ALS']
        }
    }
    
    for mode, info in tasks.items():
        print(f"模式: {mode}")
        print(f"  描述: {info['description']}")
        print(f"  适用任务: {', '.join(info['suitable_for'])}")
        print(f"  模型类型: {', '.join(info['model_types'])}")
        print()
    
    print()


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("张量到矩阵转换工具 - 使用示例")
    print("="*70 + "\n")
    
    # 运行示例
    example_1_load_and_inspect()
    example_2_train_simple_model()
    example_3_convert_new_dataset()
    example_4_comparison_with_original()
    example_5_different_modes_for_different_tasks()
    
    print("="*70)
    print("所有示例运行完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
