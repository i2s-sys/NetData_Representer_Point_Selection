"""
基于现有数据加载流程的张量转换工具
参考 config.py 和 utils.py 的数据读取方式
"""

import numpy as np
import configparser
import os


class TensorConverter:
    """张量转换器，兼容现有数据加载格式"""
    
    def __init__(self, dataset_name: str, data_dir: str = './data'):
        """
        初始化转换器
        
        参数:
            dataset_name: 数据集名称（如 'Abilene'）
            data_dir: 数据目录
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.tensor = None
        self.ndim = None
        self.config = None
        
        self._load_config()
        self._load_tensor()
    
    def _load_config(self):
        """加载配置文件（参考 config.py 的加载方式）"""
        config_file = os.path.join(self.data_dir, f"{self.dataset_name}.ini")
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        conf = configparser.ConfigParser()
        conf.read(config_file)
        
        # 读取维度信息
        self.ndim = np.array(json.loads(conf.get("Data_Setting", "ndim")))
        self.config = conf
        
        print(f"[配置加载] 数据集: {self.dataset_name}")
        print(f"[配置加载] 维度 ndim: {self.ndim}")
    
    def _load_tensor(self):
        """加载张量数据"""
        tensor_path = os.path.join(self.data_dir, f"{self.dataset_name}.npy")
        
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"张量文件不存在: {tensor_path}")
        
        self.tensor = np.load(tensor_path)
        print(f"[张量加载] 形状: {self.tensor.shape}, dtype: {self.tensor.dtype}")
    
    def convert_to_matrix(self, mode: str = 'flatten', save: bool = True, output_dir: str = './handleData/output'):
        """
        将张量转换为矩阵

        注意：数据是网络流量数据，维度为 (源节点, 目标节点, 时间)

        参数:
            mode: 转换模式
                - 'flatten': (source, dest, time) -> (source*dest, time)
                - 'source_time': (source, dest, time) -> (source, dest*time)
                - 'dest_time': (source, dest, time) -> (dest, source*time)
                - 'time_link': (source, dest, time) -> (time, source*dest) - 每列是一条链路
                - 'avg': (source, dest, time) -> (source, dest) 按时间平均
            save: 是否保存到文件
            output_dir: 输出目录

        返回:
            转换后的矩阵
        """
        num_source, num_dest, num_time = self.tensor.shape

        if mode == 'flatten':
            # 展平前两维: (source, dest, time) -> (source*dest, time)
            matrix = self.tensor.reshape(num_source * num_dest, num_time)
            desc = f"({num_source}×{num_dest}, {num_time})"

        elif mode == 'source_time' or mode == 'user_time':
            # 按源节点展开: (source, dest, time) -> (source, dest*time)
            matrix = self.tensor.reshape(num_source, num_dest * num_time)
            desc = f"({num_source}, {num_dest}×{num_time})"

        elif mode == 'dest_time' or mode == 'item_time':
            # 按目标节点展开: (source, dest, time) -> (dest, source*time)
            matrix = self.tensor.transpose(1, 0, 2).reshape(num_dest, num_source * num_time)
            desc = f"({num_dest}, {num_source}×{num_time})"

        elif mode == 'time_link' or mode == 'time_user_item':
            # 按时间展开: (source, dest, time) -> (time, source*dest)
            matrix = self.tensor.transpose(2, 0, 1).reshape(num_time, num_source * num_dest)
            desc = f"({num_time}, {num_source}×{num_dest})"

        elif mode == 'avg':
            # 按时间平均: (source, dest, time) -> (source, dest)
            matrix = self.tensor.mean(axis=2)
            desc = f"({num_source}, {num_dest})"

        else:
            raise ValueError(f"未知的转换模式: {mode}")
        
        print(f"[转换完成] 模式: {mode}, 输出形状: {matrix.shape}")
        
        if save:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.dataset_name}_matrix_{mode}.npy")
            np.save(output_path, matrix)
            print(f"[保存] 输出文件: {output_path}")
        
        return matrix
    
    def convert_all_modes(self, output_dir: str = './handleData/output'):
        """
        转换所有模式并保存
        
        参数:
            output_dir: 输出目录
        """
        modes = ['flatten', 'user_time', 'item_time', 'time_user_item', 'avg']
        
        print(f"\n{'='*70}")
        print(f"开始批量转换数据集: {self.dataset_name}")
        print(f"{'='*70}\n")
        
        for mode in modes:
            try:
                self.convert_to_matrix(mode=mode, save=True, output_dir=output_dir)
            except Exception as e:
                print(f"[错误] 转换模式 {mode} 失败: {e}")
        
        print(f"\n{'='*70}")
        print(f"批量转换完成！输出目录: {output_dir}")
        print(f"{'='*70}\n")


def convert_single_sample(dataset_name: str, mode: str = 'flatten'):
    """
    简单的单次转换示例
    
    参数:
        dataset_name: 数据集名称
        mode: 转换模式
    """
    print(f"正在转换: {dataset_name} ({mode})")
    converter = TensorConverter(dataset_name)
    matrix = converter.convert_to_matrix(mode=mode, save=True)
    return matrix


def convert_all_datasets(data_dir: str = './data', output_dir: str = './handleData/output'):
    """
    转换所有可用的数据集
    
    参数:
        data_dir: 数据目录
        output_dir: 输出目录
    """
    # 查找所有 .ini 配置文件
    config_files = [f for f in os.listdir(data_dir) if f.endswith('.ini')]
    
    print(f"\n{'='*70}")
    print(f"发现 {len(config_files)} 个数据集配置文件")
    print(f"{'='*70}\n")
    
    for config_file in config_files:
        dataset_name = config_file.replace('.ini', '')
        print(f"\n处理数据集: {dataset_name}")
        
        try:
            converter = TensorConverter(dataset_name, data_dir)
            converter.convert_all_modes(output_dir)
        except Exception as e:
            print(f"[错误] 转换数据集 {dataset_name} 失败: {e}")


def load_converted_matrix(dataset_name: str, mode: str = 'flatten', 
                          data_dir: str = './handleData/output'):
    """
    加载已转换的矩阵
    
    参数:
        dataset_name: 数据集名称
        mode: 转换模式
        data_dir: 矩阵文件所在目录
    
    返回:
        矩阵数据
    """
    matrix_path = os.path.join(data_dir, f"{dataset_name}_matrix_{mode}.npy")
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"矩阵文件不存在: {matrix_path}")
    
    matrix = np.load(matrix_path)
    print(f"[加载] 矩阵: {matrix_path}, 形状: {matrix.shape}")
    
    return matrix


if __name__ == "__main__":
    import json
    
    print("\n" + "="*70)
    print("张量转换工具 (基于现有数据加载格式)")
    print("="*70 + "\n")
    
    # 示例1: 转换单个数据集
    print("示例1: 转换 Abilene 数据集 (所有模式)")
    converter = TensorConverter('Abilene')
    converter.convert_all_modes()
    
    # 示例2: 只转换特定模式
    print("\n示例2: 只转换 flatten 模式")
    converter2 = TensorConverter('Abilene')
    matrix = converter2.convert_to_matrix(mode='flatten', output_dir='./handleData/output')
    print(f"矩阵形状: {matrix.shape}")
    
    # 示例3: 加载已转换的矩阵
    print("\n示例3: 加载已转换的矩阵")
    try:
        loaded_matrix = load_converted_matrix('Abilene', mode='flatten')
        print(f"成功加载矩阵，形状: {loaded_matrix.shape}")
    except FileNotFoundError as e:
        print(f"未找到矩阵文件: {e}")
