"""
将网络流量张量转换为二维矩阵（每一列表示一个时间点）
输出形状: (3000, 144) - (时间点, 源节点×目标节点)

使用方法:
    cd handleData
    python convert_network_tensor.py

或指定数据集:
    python convert_network_tensor.py --dataset Abilene
"""

import numpy as np
import argparse
import os
import sys


def convert_tensor_to_matrix(dataset_name='Abilene', data_dir='../data', output_dir='./output'):
    """
    将网络流量张量转换为矩阵

    参数:
        dataset_name: 数据集名称 (Abilene, Geant, etc.)
        data_dir: 数据目录（相对于本脚本的位置）
        output_dir: 输出目录
    """
    print("="*70)
    print(f"转换 {dataset_name} 网络流量张量为二维矩阵（每列=一个时间点）")
    print("="*70 + "\n")

    # 加载张量
    tensor_file = os.path.join(data_dir, f"{dataset_name}.npy")

    if not os.path.exists(tensor_file):
        print(f"错误：找不到文件 {tensor_file}")
        print(f"当前目录: {os.getcwd()}")
        print(f"请确保路径正确")
        return None

    print(f"正在加载张量: {tensor_file}")
    tensor = np.load(tensor_file)
    print(f"原始张量形状: {tensor.shape}")

    # 从张量获取维度
    num_source, num_dest, num_time = tensor.shape

    print(f"\n数据集信息:")
    print(f"  - 源节点数量: {num_source}")
    print(f"  - 目标节点数量: {num_dest}")
    print(f"  - 时间点数量: {num_time}")
    print(f"  - 总链路数: {num_source * num_dest} (源节点×目标节点)")
    print()

    # 转换为矩阵（每列=一个时间点）
    # (source, destination, time) -> (time, source * destination)
    print("正在转换...")
    matrix = tensor.transpose(2, 0, 1).reshape(num_time, num_source * num_dest)

    print(f"\n转换完成！")
    print(f"输出矩阵形状: {matrix.shape}")
    print(f"  - 行数（时间点）: {matrix.shape[0]}")
    print(f"  - 列数（源节点×目标节点/链路数）: {matrix.shape[1]}")
    print(f"\n每一列代表一条网络链路（源节点->目标节点）")
    print(f"每一行代表特定时间点的网络状态快照")

    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print(f"  均值: {matrix.mean():.4f}")
    print(f"  标准差: {matrix.std():.4f}")
    print(f"  中位数: {np.median(matrix):.4f}")
    print(f"  非零值比例: {(matrix != 0).sum() / matrix.size:.2%}")
    print(f"  零值数量: {(matrix == 0).sum()}")
    print(f"  总元素数: {matrix.size}")

    # 保存矩阵
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}_matrix_time_link.npy')

    np.save(output_path, matrix)
    print(f"\n输出文件已保存到: {output_path}")

    # 验证保存的文件
    verify_matrix = np.load(output_path)
    if np.array_equal(matrix, verify_matrix):
        print("✓ 验证成功：保存的矩阵数据一致")
    else:
        print("✗ 警告：保存的矩阵数据不一致")

    print("="*70)

    # 显示前几行和列的示例数据
    print(f"\n示例数据（前5行，前10列）:")
    print(f"{'':^8}", end='')
    for j in range(min(10, matrix.shape[1])):
        print(f"{f'链路{j}':^12}", end='')
    print()
    for i in range(min(5, matrix.shape[0])):
        print(f"时间{i:^5}", end=' ')
        for j in range(min(10, matrix.shape[1])):
            print(f"{matrix[i, j]:^12.2f}", end='')
        print()

    return matrix


def main():
    parser = argparse.ArgumentParser(description='将网络流量张量转换为二维矩阵')
    parser.add_argument('--dataset', type=str, default='Abilene',
                        help='数据集名称 (默认: Abilene)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录 (默认: ../data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')

    args = parser.parse_args()

    # 执行转换
    convert_tensor_to_matrix(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
