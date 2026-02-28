"""
快速检查数据文件的实际内容和格式
"""

import numpy as np
import os

print("检查数据文件...")
print("="*70)

# 检查文件是否存在
data_dir = '../data'
dataset = 'Abilene'

npy_file = os.path.join(data_dir, f'{dataset}.npy')
npz_file = os.path.join(data_dir, f'{dataset}.npz')

print(f"当前目录: {os.getcwd()}")
print(f"\n查找的文件:")
print(f"  {npy_file}")
print(f"  {npz_file}")

# 检查文件
for filepath in [npy_file, npz_file]:
    if os.path.exists(filepath):
        print(f"\n✓ 找到文件: {os.path.basename(filepath)}")
        print(f"  文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

        try:
            # 尝试加载
            data = np.load(filepath, allow_pickle=True)
            print(f"  ✓ 加载成功")

            if isinstance(data, np.ndarray):
                print(f"  数据类型: numpy.ndarray")
                print(f"  形状: {data.shape}")
                print(f"  dtype: {data.dtype}")
                print(f"  总大小: {data.size}")

                # 计算预期大小
                expected_size = 12 * 12 * 3000
                print(f"\n  预期大小（从配置）: {expected_size} (12 × 12 × 3000)")
                print(f"  实际大小: {data.size}")

                if data.size == expected_size:
                    print(f"  ✓ 大小匹配！")
                else:
                    print(f"  ✗ 大小不匹配！差异: {data.size - expected_size}")
                    print(f"\n  尝试推断可能的3D形状:")

                    # 尝试找出可能的维度
                    size = data.size
                    possible_shapes = []

                    # 尝试常见的维度组合
                    for dim1 in [12, 10, 15, 20, 24]:
                        if size % dim1 == 0:
                            rem1 = size // dim1
                            for dim2 in [12, 10, 15, 20, 24, 48]:
                                if rem1 % dim2 == 0:
                                    dim3 = rem1 // dim2
                                    possible_shapes.append((dim1, dim2, dim3))

                    if possible_shapes:
                        print(f"  可能的3D形状:")
                        for shape in possible_shapes[:10]:  # 只显示前10个
                            print(f"    {shape}")
                    else:
                        print(f"  无法推断3D形状")

                    # 检查是否已经是矩阵
                    if len(data.shape) == 2:
                        print(f"\n  数据已经是2D矩阵")
                        print(f"  可能的原因:")
                        print(f"    1. 这是已经转换过的数据")
                        print(f"    2. 配置文件与实际数据不匹配")

                    # 检查是否是1D向量
                    elif len(data.shape) == 1:
                        print(f"\n  数据是1D向量，尝试reshape...")
                        for shape in [(12, 12, -1), (10, 10, -1), (15, 15, -1)]:
                            if size % (shape[0] * shape[1]) == 0:
                                dim3 = size // (shape[0] * shape[1])
                                print(f"    可以reshape为: ({shape[0]}, {shape[1]}, {dim3})")

            elif isinstance(data, np.lib.npyio.NpzFile):
                print(f"  数据类型: npz文件（压缩格式）")
                print(f"  包含的数组:")
                for key in data.keys():
                    arr = data[key]
                    print(f"    '{key}': shape={arr.shape}, dtype={arr.dtype}, size={arr.size}")
                data.close()

            else:
                print(f"  数据类型: {type(data)}")
                print(f"  内容: {data}")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ 文件不存在: {os.path.basename(filepath)}")

print("\n" + "="*70)
