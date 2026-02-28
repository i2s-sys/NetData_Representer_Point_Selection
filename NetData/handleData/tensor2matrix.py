"""
张量到矩阵转换工具
支持多种将3D张量（用户×物品×时间）转换为2D矩阵的方式
"""

import numpy as np
import configparser
import os
import json
from typing import Tuple, List


class Tensor2MatrixConverter:
    """将3D张量转换为2D矩阵"""
    
    def __init__(self, config_file: str = None, tensor_path: str = None):
        """
        初始化转换器
        
        参数:
            config_file: 配置文件路径（.ini格式）
            tensor_path: 张量文件路径（.npy格式）
        """
        self.tensor = None
        self.ndim = None  # [num_user, num_item, num_time]
        self.config = None
        
        if config_file:
            self.load_config(config_file)
            
        if tensor_path:
            self.load_tensor(tensor_path)
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
        conf = configparser.ConfigParser()
        conf.read(config_file)
        
        self.ndim = np.array(json.loads(conf.get("Data_Setting", "ndim")))
        self.config = conf
        print(f"加载配置成功: ndim = {self.ndim}")
    
    def load_tensor(self, tensor_path: str):
        """加载张量数据"""
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"张量文件不存在: {tensor_path}")
            
        self.tensor = np.load(tensor_path)
        print(f"加载张量成功，形状: {self.tensor.shape}")
        
        if self.ndim is None:
            self.ndim = np.array(self.tensor.shape)
            print(f"自动推断维度: {self.ndim}")
    
    def reshape_user_time(self) -> np.ndarray:
        """
        按用户维度展开: (user, item, time) -> (user, item * time)
        
        返回:
            矩阵: 用户 × (物品 × 时间)
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        num_user, num_item, num_time = self.tensor.shape
        matrix = self.tensor.reshape(num_user, num_item * num_time)
        print(f"按用户维度展开: {self.tensor.shape} -> {matrix.shape}")
        return matrix
    
    def reshape_item_time(self) -> np.ndarray:
        """
        按物品维度展开: (user, item, time) -> (item, user * time)
        
        返回:
            矩阵: 物品 × (用户 × 时间)
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        num_user, num_item, num_time = self.tensor.shape
        matrix = self.tensor.transpose(1, 0, 2).reshape(num_item, num_user * num_time)
        print(f"按物品维度展开: {self.tensor.shape} -> {matrix.shape}")
        return matrix
    
    def reshape_time_user_item(self) -> np.ndarray:
        """
        按时间维度展开: (user, item, time) -> (time, user * item)
        
        返回:
            矩阵: 时间 × (用户 × 物品)
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        num_user, num_item, num_time = self.tensor.shape
        matrix = self.tensor.transpose(2, 0, 1).reshape(num_time, num_user * num_item)
        print(f"按时间维度展开: {self.tensor.shape} -> {matrix.shape}")
        return matrix
    
    def reshape_user_item_avg(self) -> np.ndarray:
        """
        按用户-物品对平均: (user, item, time) -> (user * item, time_avg)
        
        返回:
            矩阵: (用户 × 物品) × 平均值（标量）
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        num_user, num_item, num_time = self.tensor.shape
        # 计算每个用户-物品对的平均值
        avg_matrix = self.tensor.mean(axis=2)
        print(f"按用户-物品对平均: {self.tensor.shape} -> {avg_matrix.shape}")
        return avg_matrix
    
    def reshape_user_item_flatten(self) -> np.ndarray:
        """
        展平前两维: (user, item, time) -> (user * item, time)
        
        返回:
            矩阵: (用户 × 物品) × 时间
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        num_user, num_item, num_time = self.tensor.shape
        matrix = self.tensor.reshape(num_user * num_item, num_time)
        print(f"展平前两维: {self.tensor.shape} -> {matrix.shape}")
        return matrix
    
    def reshape_user_item_flatten_with_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        展平前两维并保留标签: (user, item, time) -> (user * item, time)
        
        返回:
            matrix: 矩阵 (user * item, time)
            labels: 标签数组 (user * item, 2)，每行是[user_index, item_index]
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        num_user, num_item, num_time = self.tensor.shape
        matrix = self.tensor.reshape(num_user * num_item, num_time)
        
        # 生成标签
        user_indices = np.repeat(np.arange(num_user), num_item)
        item_indices = np.tile(np.arange(num_item), num_user)
        labels = np.column_stack([user_indices, item_indices])
        
        print(f"展平前两维: {self.tensor.shape} -> {matrix.shape}, 标签: {labels.shape}")
        return matrix, labels
    
    def flatten_to_2d(self, axis: int = 2) -> np.ndarray:
        """
        指定维度展开为2D
        
        参数:
            axis: 要保留的维度 (0=user, 1=item, 2=time)
        
        返回:
            2D矩阵
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        if axis == 0:
            # 保留用户维度: (user, item, time) -> (user, item * time)
            return self.reshape_user_time()
        elif axis == 1:
            # 保留物品维度: (user, item, time) -> (item, user * time)
            return self.reshape_item_time()
        elif axis == 2:
            # 保留时间维度: (user, item, time) -> (time, user * item)
            return self.reshape_time_user_item()
        else:
            raise ValueError(f"无效的axis值: {axis}，必须是0, 1, 或2")
    
    def save_matrix(self, matrix: np.ndarray, output_path: str):
        """
        保存矩阵到文件
        
        参数:
            matrix: 要保存的矩阵
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, matrix)
        print(f"矩阵已保存到: {output_path}")
    
    def convert_and_save(self, method: str = 'flatten', output_dir: str = './handleData/output'):
        """
        使用指定方法转换并保存所有可能的矩阵
        
        参数:
            method: 转换方法 ('user_time', 'item_time', 'time_user_item', 
                    'flatten', 'avg', 'all')
            output_dir: 输出目录
        """
        if self.tensor is None:
            raise ValueError("请先加载张量数据")
            
        os.makedirs(output_dir, exist_ok=True)
        
        methods = {
            'user_time': self.reshape_user_time,
            'item_time': self.reshape_item_time,
            'time_user_item': self.reshape_time_user_item,
            'flatten': self.reshape_user_item_flatten,
            'avg': self.reshape_user_item_avg
        }
        
        if method == 'all':
            for name, func in methods.items():
                matrix = func()
                output_path = os.path.join(output_dir, f"matrix_{name}.npy")
                self.save_matrix(matrix, output_path)
        elif method in methods:
            matrix = methods[method]()
            output_path = os.path.join(output_dir, f"matrix_{method}.npy")
            self.save_matrix(matrix, output_path)
        else:
            raise ValueError(f"未知的转换方法: {method}")


def convert_dataset(dataset_name: str = 'Abilene', 
                   data_dir: str = './data', 
                   output_dir: str = './handleData/output',
                   methods: List[str] = None):
    """
    便捷函数：转换指定数据集
    
    参数:
        dataset_name: 数据集名称
        data_dir: 数据目录
        output_dir: 输出目录
        methods: 转换方法列表，如果为None则使用['flatten', 'user_time', 'item_time', 'time_user_item']
    """
    if methods is None:
        methods = ['flatten', 'user_time', 'item_time', 'time_user_item']
    
    # 加载配置
    config_file = os.path.join(data_dir, f"{dataset_name}.ini")
    tensor_file = os.path.join(data_dir, f"{dataset_name}.npy")
    
    print(f"\n{'='*60}")
    print(f"开始转换数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    # 创建转换器
    converter = Tensor2MatrixConverter(config_file, tensor_file)
    
    # 执行转换
    for method in methods:
        print(f"\n使用方法: {method}")
        try:
            converter.convert_and_save(method, output_dir)
        except Exception as e:
            print(f"转换失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"转换完成！输出目录: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import json
    
    print("张量到矩阵转换工具\n")
    
    # 示例：转换 Abilene 数据集
    dataset_name = 'Abilene'
    
    print(f"正在转换数据集: {dataset_name}")
    convert_dataset(
        dataset_name=dataset_name,
        data_dir='./data',
        output_dir='./handleData/output',
        methods=['flatten', 'user_time', 'item_time', 'time_user_item', 'avg']
    )
