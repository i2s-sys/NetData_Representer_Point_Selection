"""
批量转换多个数据集为矩阵（列是时间点）

使用方法:
    cd handleData
    python convert_batch.py

或指定数据集:
    python convert_batch.py --datasets Abilene_12_12_3000,Geant_23_23_3000
"""

import numpy as np
import argparse
import os
import sys


def convert_dataset(dataset_name, data_dir='../data', output_dir='./output'):
    """转换单个数据集"""
    print("\n" + "="*70)
    print(f"转换数据集: {dataset_name}")
    print("="*70)

    # 加载数据
    tensor_file = os.path.join(data_dir, f"{dataset_name}.npy")

    if not os.path.exists(tensor_file):
        print(f"✗ 跳过：找不到文件 {tensor_file}")
        return False

    print(f"正在加载文件: {tensor_file}")
    sparse_data = np.load(tensor_file)
    print(f"✓ 加载成功")
    print(f"  数据形状: {sparse_data.shape}")

    # 检查数据格式
    if len(sparse_data.shape) != 2 or sparse_data.shape[1] != 4:
        print(f"✗ 跳过：不支持的数据格式")
        print(f"  期望: (N, 4) - [源节点, 目标节点, 时间, 流量]")
        print(f"  实际: {sparse_data.shape}")
        return False

    # 提取各列
    sources = sparse_data[:, 0].astype(int)
    dests = sparse_data[:, 1].astype(int)
    times = sparse_data[:, 2].astype(int)
    values = sparse_data[:, 3]

    # 确定维度
    num_source = sources.max() + 1
    num_dest = dests.max() + 1
    num_time = times.max() + 1
    num_link = num_source * num_dest

    print(f"\n数据集维度:")
    print(f"  源节点数: {num_source}")
    print(f"  目标节点数: {num_dest}")
    print(f"  时间点数: {num_time}")
    print(f"  总链路数: {num_link}")
    print(f"  数据点数: {len(sources)}")

    # 转换为矩阵（列是时间点）
    # 输出形状: (link_count, time_count)
    print(f"\n开始转换...")
    print(f"目标形状: (链路数, 时间点) = ({num_link}, {num_time})")

    matrix = np.zeros((num_link, num_time), dtype=values.dtype)

    # 填充矩阵
    print(f"  填充 {len(sources)} 个数据点...")
    for i, (s, d, t, v) in enumerate(zip(sources, dests, times, values)):
        row = s * num_dest + d
        matrix[row, t] = v

        if (i + 1) % 100000 == 0:
            print(f"    进度: {i+1}/{len(sources)}")

    print(f"  ✓ 填充完成")

    print(f"\n转换完成！")
    print(f"  输出形状: {matrix.shape}")
    print(f"  - 行数（链路数）: {matrix.shape[0]}")
    print(f"  - 列数（时间点）: {matrix.shape[1]}")

    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print(f"  均值: {matrix.mean():.4f}")
    print(f"  非零值比例: {(matrix != 0).sum() / matrix.size:.2%}")

    # 保存矩阵
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}_matrix.npy')

    np.save(output_path, matrix)
    print(f"\n✓ 已保存到: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='批量转换数据集')
    parser.add_argument('--datasets', type=str,
                        default='Abilene_12_12_3000,Geant_23_23_3000',
                        help='数据集列表，用逗号分隔')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录 (默认: ../data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')

    args = parser.parse_args()

    # 解析数据集列表
    dataset_list = [d.strip() for d in args.datasets.split(',')]

    print("="*70)
    print(f"批量转换 {len(dataset_list)} 个数据集")
    print("="*70)
    print(f"数据集列表: {dataset_list}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")

    # 逐个转换
    success_count = 0
    failed_count = 0

    for dataset_name in dataset_list:
        if convert_dataset(dataset_name, args.data_dir, args.output_dir):
            success_count += 1
        else:
            failed_count += 1

    # 总结
    print("\n" + "="*70)
    print("批量转换完成")
    print("="*70)
    print(f"成功: {success_count}/{len(dataset_list)}")
    print(f"失败: {failed_count}/{len(dataset_list)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
