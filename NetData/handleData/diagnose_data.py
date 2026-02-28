"""
诊断数据文件结构和内容
"""

import numpy as np
import os


def diagnose_file(filepath):
    """诊断文件类型和内容"""
    print("="*70)
    print(f"诊断文件: {filepath}")
    print("="*70)

    if not os.path.exists(filepath):
        print(f"✗ 文件不存在: {filepath}")
        return None

    print(f"✓ 文件存在")
    print(f"  文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

    # 检查文件扩展名
    ext = os.path.splitext(filepath)[1].lower()
    print(f"  文件扩展名: {ext}")

    try:
        # 尝试加载文件
        data = np.load(filepath, allow_pickle=True)

        # 检查加载的数据类型
        print(f"\n✓ 文件加载成功")

        if isinstance(data, np.ndarray):
            print(f"  数据类型: numpy.ndarray")
            print(f"  形状: {data.shape}")
            print(f"  数据类型(dtype): {data.dtype}")
            print(f"  总元素数: {data.size}")
            print(f"  范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  均值: {data.mean():.4f}")
            print(f"  是否包含NaN: {np.isnan(data).any()}")
            print(f"  是否包含Inf: {np.isinf(data).any()}")
            print(f"  非零元素数: {(data != 0).sum()}")
            print(f"  非零比例: {(data != 0).sum() / data.size:.2%}")
            return data

        elif isinstance(data, np.lib.npyio.NpzFile):
            print(f"  数据类型: npz文件（压缩的多个数组）")
            print(f"  包含的数组:")
            for key in data.keys():
                arr = data[key]
                print(f"    - '{key}': shape={arr.shape}, dtype={arr.dtype}, size={arr.size}")
            return data

        else:
            print(f"  数据类型: {type(data)}")
            print(f"  数据内容: {data}")
            return data

    except Exception as e:
        print(f"✗ 加载文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 检查 Abilene 数据文件
    data_dir = '../data'
    dataset_name = 'Abilene'

    print(f"\n当前工作目录: {os.getcwd()}")
    print(f"数据目录: {data_dir}")
    print(f"数据目录存在: {os.path.exists(data_dir)}")

    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"数据目录中的文件: {len(files)} 个")
        for f in sorted(files):
            if dataset_name.lower() in f.lower():
                print(f"  - {f}")

    print("\n" + "="*70)

    # 诊断 .npy 文件
    npy_file = os.path.join(data_dir, f"{dataset_name}.npy")
    npy_data = diagnose_file(npy_file)

    print("\n" + "="*70)

    # 诊断 .npz 文件（如果存在）
    npz_file = os.path.join(data_dir, f"{dataset_name}.npz")
    if os.path.exists(npz_file):
        npz_data = diagnose_file(npz_file)
    else:
        print(f"\n没有找到 .npz 文件: {npz_file}")

    print("\n" + "="*70)

    # 分析问题
    print("\n问题分析:")
    if npy_data is not None and isinstance(npy_data, np.ndarray):
        expected_size = 12 * 12 * 3000  # 配置中的维度
        actual_size = npy_data.size

        print(f"  预期大小（从配置）: {expected_size} (12 × 12 × 3000)")
        print(f"  实际大小（从文件）: {actual_size}")

        if actual_size == expected_size:
            print(f"  ✓ 大小匹配！")
        else:
            print(f"  ✗ 大小不匹配！差异: {expected_size - actual_size}")
            print(f"\n  可能的原因:")
            print(f"    1. 文件和配置不匹配")
            print(f"    2. 数据集版本不同")
            print(f"    3. 数据已被处理或修改")

            # 尝试找出可能的维度
            print(f"\n  尝试推断可能的维度:")

            # 检查是否能整除
            for dim1 in [12, 10, 20, 15]:
                if actual_size % dim1 == 0:
                    remainder = actual_size // dim1
                    for dim2 in [12, 10, 20, 15, 48]:
                        if remainder % dim2 == 0:
                            dim3 = remainder // dim2
                            print(f"    可能的形状: ({dim1}, {dim2}, {dim3})")

        # 尝试 reshape
        print(f"\n  尝试 reshape:")
        try:
            reshaped = npy_data.reshape(12, 12, -1)
            print(f"    ✓ reshape(12, 12, -1) 成功: {reshaped.shape}")
        except Exception as e:
            print(f"    ✗ reshape(12, 12, -1) 失败: {e}")

        try:
            reshaped = npy_data.reshape(-1, 12, 12)
            print(f"    ✓ reshape(-1, 12, 12) 成功: {reshaped.shape}")
        except Exception as e:
            print(f"    ✗ reshape(-1, 12, 12) 失败: {e}")

    print("\n" + "="*70)
    print("诊断完成")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
