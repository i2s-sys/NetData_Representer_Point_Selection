"""
将稀疏索引格式转换为矩阵（列是时间点）

输入格式: (N, 4) - [源节点, 目标节点, 时间, 流量值]
输出格式: (链路数, 时间点) - 每一列是一个时间点

使用方法:
    cd handleData
    python convert_to_matrix_col_time.py --dataset Abilene_12_12_3000
"""

import numpy as np
import argparse
import os


def convert_sparse_to_matrix_col_time(dataset_name, data_dir='../data', output_dir='./output'):
    """
    将稀疏索引格式转换为矩阵（列是时间点）

    输入: (N, 4) - [source, dest, time, value]
    输出: (link_count, time_count) - 每一列是一个时间点
    """
    print("="*70)
    print(f"转换 {dataset_name} 为矩阵（列=时间点）")
    print("="*70 + "\n")

    # 加载数据
    tensor_file = os.path.join(data_dir, f"{dataset_name}.npy")

    if not os.path.exists(tensor_file):
        print(f"✗ 错误：找不到文件 {tensor_file}")
        return None

    print(f"正在加载文件: {tensor_file}")
    sparse_data = np.load(tensor_file)
    print(f"✓ 加载成功")
    print(f"  数据形状: {sparse_data.shape}")
    print(f"  数据类型: {sparse_data.dtype}")

    # 检查数据格式
    if sparse_data.shape[1] != 4:
        print(f"\n✗ 错误：期望4列 [源节点, 目标节点, 时间, 值]")
        print(f"  实际列数: {sparse_data.shape[1]}")
        return None

    # 提取各列
    sources = sparse_data[:, 0].astype(int)
    dests = sparse_data[:, 1].astype(int)
    times = sparse_data[:, 2].astype(int)
    values = sparse_data[:, 3]

    print(f"\n数据列分析:")
    print(f"  源节点索引范围: [{sources.min()}, {sources.max()}]")
    print(f"  目标节点索引范围: [{dests.min()}, {dests.max()}]")
    print(f"  时间索引范围: [{times.min()}, {times.max()}]")
    print(f"  流量值范围: [{values.min():.4f}, {values.max():.4f}]")

    # 确定维度
    num_source = sources.max() + 1
    num_dest = dests.max() + 1
    num_time = times.max() + 1
    num_link = num_source * num_dest

    print(f"\n推断的维度:")
    print(f"  源节点数: {num_source}")
    print(f"  目标节点数: {num_dest}")
    print(f"  时间点数: {num_time}")
    print(f"  总链路数: {num_link}")

    # 验证维度
    expected_points = num_source * num_dest * num_time
    actual_points = len(sources)

    print(f"\n数据验证:")
    print(f"  预期数据点数: {expected_points}")
    print(f"  实际数据点数: {actual_points}")
    print(f"  稀疏度: {(1 - actual_points/expected_points)*100:.2f}%")

    # 转换为矩阵（列是时间点）
    # 输出形状: (link_count, time_count)
    print(f"\n开始转换...")
    print(f"目标形状: (链路数, 时间点) = ({num_link}, {num_time})")
    print(f"  每一行 = 一条链路")
    print(f"  每一列 = 一个时间点")

    # 初始化矩阵: (链路数, 时间点)
    matrix = np.zeros((num_link, num_time), dtype=values.dtype)

    # 填充矩阵
    print(f"  填充 {len(sources)} 个数据点...")
    for i, (s, d, t, v) in enumerate(zip(sources, dests, times, values)):
        row = s * num_dest + d  # 链路索引: 源节点 * 目标节点数 + 目标节点
        col = t  # 时间索引
        matrix[row, col] = v

        if (i + 1) % 100000 == 0:
            print(f"    进度: {i+1}/{len(sources)}")

    print(f"  ✓ 填充完成")

    print(f"\n转换完成！")
    print(f"输出矩阵形状: {matrix.shape}")
    print(f"  - 行数（链路数）: {matrix.shape[0]}")
    print(f"  - 列数（时间点）: {matrix.shape[1]}")
    print(f"\n每一行代表一条网络链路（源节点->目标节点）")
    print(f"每一列代表一个时间点的网络状态")

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
    output_path = os.path.join(output_dir, f'{dataset_name}_matrix.npy')

    np.save(output_path, matrix)
    print(f"\n✓ 已保存到: {output_path}")

    # 验证
    verify = np.load(output_path)
    if np.array_equal(matrix, verify):
        print(f"✓ 验证成功：数据一致")
    else:
        print(f"✗ 警告：数据不一致")

    print("="*70)

    # 显示前几行和列的示例数据
    print(f"\n示例数据（前5行链路，前10列时间点）:")
    print(f"{'':^12}", end='')
    for j in range(min(10, matrix.shape[1])):
        print(f"{f'时间{j}':^10}", end='')
    print()
    for i in range(min(5, matrix.shape[0])):
        link_src = i // num_dest
        link_dst = i % num_dest
        print(f"链路{i:2d}({link_src}->{link_dst})", end=' ')
        for j in range(min(10, matrix.shape[1])):
            print(f"{matrix[i, j]:^10.2f}", end='')
        print()

    return matrix


def main():
    parser = argparse.ArgumentParser(description='将稀疏格式转换为矩阵（列=时间点）')
    parser.add_argument('--dataset', type=str, default='Abilene_12_12_3000',
                        help='数据集名称 (默认: Abilene_12_12_3000)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录 (默认: ../data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')

    args = parser.parse_args()

    convert_sparse_to_matrix_col_time(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
