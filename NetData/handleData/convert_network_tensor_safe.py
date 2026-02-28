"""
安全地转换网络流量张量为矩阵
自动检测文件类型和数据格式

使用方法:
    cd handleData
    python convert_network_tensor_safe.py
"""

import numpy as np
import os
import sys


class SafeTensorConverter:
    """安全的张量转换器，处理各种数据格式"""

    def __init__(self, dataset_name, data_dir='../data'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.tensor = None
        self.actual_shape = None

    def load_tensor(self):
        """安全地加载张量，自动检测文件类型"""
        print("="*70)
        print(f"加载 {self.dataset_name} 张量")
        print("="*70 + "\n")

        # 尝试不同的文件扩展名
        possible_files = [
            f"{self.dataset_name}.npy",
            f"{self.dataset_name}.npz",
            f"{self.dataset_name.lower()}.npy",
            f"{self.dataset_name.lower()}.npz",
        ]

        data_path = None
        for filename in possible_files:
            full_path = os.path.join(self.data_dir, filename)
            if os.path.exists(full_path):
                data_path = full_path
                print(f"找到文件: {filename}")
                break

        if data_path is None:
            print(f"✗ 错误：找不到 {self.dataset_name} 数据文件")
            print(f"尝试过的文件:")
            for f in possible_files:
                print(f"  - {os.path.join(self.data_dir, f)}")
            return False

        print(f"完整路径: {data_path}")
        print(f"文件大小: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB\n")

        try:
            # 加载文件
            data = np.load(data_path, allow_pickle=True)

            # 处理不同类型的数据
            if isinstance(data, np.ndarray):
                print(f"✓ 加载成功：numpy.ndarray")
                self.tensor = data
                self.actual_shape = data.shape
                self._analyze_tensor()

            elif isinstance(data, np.lib.npyio.NpzFile):
                print(f"✓ 加载成功：npz文件（包含多个数组）")
                print(f"  数组列表:")
                for key in sorted(data.keys()):
                    arr = data[key]
                    print(f"    - '{key}': shape={arr.shape}, dtype={arr.dtype}")

                # 尝试找主数据数组
                for key in ['data', 'arr_0', 'tensor', 'values']:
                    if key in data:
                        self.tensor = data[key]
                        self.actual_shape = self.tensor.shape
                        print(f"\n  使用数组: '{key}'")
                        self._analyze_tensor()
                        break
                else:
                    # 如果没有找到预期的键，使用第一个数组
                    keys = list(data.keys())
                    self.tensor = data[keys[0]]
                    self.actual_shape = self.tensor.shape
                    print(f"\n  使用第一个数组: '{keys[0]}'")
                    self._analyze_tensor()

                data.close()

            else:
                print(f"✗ 未知的数据类型: {type(data)}")
                print(f"  数据内容: {data}")
                return False

            return True

        except Exception as e:
            print(f"✗ 加载文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _analyze_tensor(self):
        """分析张量属性"""
        print(f"\n张量属性:")
        print(f"  形状: {self.actual_shape}")
        print(f"  数据类型(dtype): {self.tensor.dtype}")
        print(f"  总元素数: {self.tensor.size}")
        print(f"  范围: [{self.tensor.min():.4f}, {self.tensor.max():.4f}]")
        print(f"  均值: {self.tensor.mean():.4f}")
        print(f"  标准差: {self.tensor.std():.4f}")
        print(f"  非零元素数: {(self.tensor != 0).sum()}")
        print(f"  非零比例: {(self.tensor != 0).sum() / self.tensor.size:.2%}")
        print(f"  是否包含NaN: {np.isnan(self.tensor).any()}")
        print(f"  是否包含Inf: {np.isinf(self.tensor).any()}")

    def detect_dimensions(self):
        """检测或推断维度"""
        print(f"\n{'='*70}")
        print(f"维度检测")
        print(f"{'='*70}\n")

        if len(self.actual_shape) == 3:
            num_source, num_dest, num_time = self.actual_shape
            print(f"✓ 检测到3D张量")
            print(f"  源节点数: {num_source}")
            print(f"  目标节点数: {num_dest}")
            print(f"  时间点数: {num_time}")
            print(f"  总链路数: {num_source * num_dest}")
            return num_source, num_dest, num_time

        elif len(self.actual_shape) == 2:
            rows, cols = self.actual_shape
            print(f"✓ 检测到2D矩阵")
            print(f"  行数: {rows}")
            print(f"  列数: {cols}")
            print(f"\n  可能的解释:")
            print(f"    1. 已经是转换后的矩阵")
            print(f"    2. 源节点×目标节点 (如果列数是时间步数)")
            return None, None, None

        elif len(self.actual_shape) == 1:
            size = self.actual_shape[0]
            print(f"✓ 检测到1D向量")
            print(f"  大小: {size}")
            print(f"\n  尝试 reshape 为3D:")

            # 尝试不同的 reshape 组合
            possible_shapes = [
                (12, 12, 3000),
                (12, 12, size // 144),
                (10, 10, size // 100),
                (20, 20, size // 400),
            ]

            for shape in possible_shapes:
                if shape[0] * shape[1] * shape[2] == size:
                    print(f"    ✓ 可以 reshape 为: {shape}")
                    self.tensor = self.tensor.reshape(shape)
                    self.actual_shape = shape
                    return self.detect_dimensions()

            print(f"  ✗ 无法推断3D形状")

        else:
            print(f"✗ 未知的维度: {len(self.actual_shape)}D")

        return None, None, None

    def convert_to_matrix(self, output_dir='./output'):
        """转换为矩阵"""
        if self.tensor is None:
            print("✗ 错误：没有加载张量数据")
            return None

        num_source, num_dest, num_time = self.detect_dimensions()

        if num_source is None:
            print("\n✗ 错误：无法确定维度，无法转换")
            return None

        print(f"\n{'='*70}")
        print(f"开始转换")
        print(f"{'='*70}\n")

        # 转换为矩阵: (source, dest, time) -> (time, source * dest)
        print(f"转换方法: (source, dest, time) -> (time, source * dest)")
        matrix = self.tensor.transpose(2, 0, 1).reshape(num_time, num_source * num_dest)

        print(f"\n转换完成！")
        print(f"  输出形状: {matrix.shape}")
        print(f"  行数（时间点）: {matrix.shape[0]}")
        print(f"  列数（链路数）: {matrix.shape[1]}")

        # 保存矩阵
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{self.dataset_name}_matrix_time_link.npy')

        np.save(output_path, matrix)
        print(f"\n✓ 已保存到: {output_path}")

        # 验证
        verify = np.load(output_path)
        if np.array_equal(matrix, verify):
            print(f"✓ 验证成功：数据一致")
        else:
            print(f"✗ 警告：数据不一致")

        return matrix


def main():
    # 获取数据集名称
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = 'Abilene'

    print(f"\n安全张量转换器")
    print(f"数据集: {dataset_name}\n")

    # 创建转换器
    converter = SafeTensorConverter(dataset_name)

    # 加载张量
    if not converter.load_tensor():
        print(f"\n✗ 转换失败：无法加载张量")
        return

    # 转换为矩阵
    matrix = converter.convert_to_matrix()

    if matrix is not None:
        print(f"\n{'='*70}")
        print(f"✓ 转换成功！")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"✗ 转换失败")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
