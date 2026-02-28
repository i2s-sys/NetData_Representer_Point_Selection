"""
将网络流量数据转换为二维矩阵
支持两种数据格式和两种输出格式：

数据格式:
1. 3D张量格式: (源节点, 目标节点, 时间)
2. 稀疏索引格式: (N, 4) - [源节点, 目标节点, 时间, 流量值]

输出格式（通过 --time_axis 参数控制）:
- --time_axis row: (时间, 链路) - 每一行=一个时间点
- --time_axis col: (链路, 时间) - 每一列=一个时间点 ⭐

使用方法:
    cd handleData
    python convert_robust.py

或指定数据集:
    python convert_robust.py --dataset Abilene_12_12_3000 --time_axis col
"""

import numpy as np
import argparse
import os


def convert_3d_to_matrix(tensor, num_source, num_dest, num_time, time_axis='row'):
    """将3D张量转换为矩阵"""
    if time_axis == 'row':
        # (source, dest, time) -> (time, source * dest)
        matrix = tensor.transpose(2, 0, 1).reshape(num_time, num_source * num_dest)
    else:  # col
        # (source, dest, time) -> (source * dest, time)
        matrix = tensor.reshape(num_source * num_dest, num_time)
    return matrix


def convert_sparse_to_matrix(sparse_data, num_source, num_dest, num_time, time_axis='row'):
    """将稀疏索引格式转换为矩阵"""
    # 提取各列
    sources = sparse_data[:, 0].astype(int)
    dests = sparse_data[:, 1].astype(int)
    times = sparse_data[:, 2].astype(int)
    values = sparse_data[:, 3]

    num_link = num_source * num_dest

    if time_axis == 'row':
        # 输出: (time, link)
        matrix = np.zeros((num_time, num_link), dtype=values.dtype)

        print(f"  填充 {len(sources)} 个数据点到 (时间, 链路) 格式...")
        for i, (s, d, t, v) in enumerate(zip(sources, dests, times, values)):
            col = s * num_dest + d  # 链路索引
            matrix[t, col] = v

    else:  # col
        # 输出: (link, time)
        matrix = np.zeros((num_link, num_time), dtype=values.dtype)

        print(f"  填充 {len(sources)} 个数据点到 (链路, 时间) 格式...")
        for i, (s, d, t, v) in enumerate(zip(sources, dests, times, values)):
            row = s * num_dest + d  # 链路索引
            matrix[row, t] = v

    if (i + 1) % 100000 == 0:
        print(f"    进度: {i+1}/{len(sources)}")

    print(f"  ✓ 填充完成")
    return matrix


def main():
    parser = argparse.ArgumentParser(description='将网络流量数据转换为二维矩阵')
    parser.add_argument('--dataset', type=str, default='Abilene_12_12_3000',
                        help='数据集名称 (默认: Abilene_12_12_3000)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录 (默认: ../data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')
    parser.add_argument('--time_axis', type=str, default='col',
                        choices=['row', 'col'],
                        help='时间轴方向: row=时间在行, col=时间在列 (默认: col)')

    args = parser.parse_args()

    print("="*70)
    print(f"转换 {args.dataset} 网络流量数据为二维矩阵")
    print("="*70)

    if args.time_axis == 'row':
        print("输出格式: (时间, 链路) - 每一行=一个时间点")
    else:
        print("输出格式: (链路, 时间) - 每一列=一个时间点 ⭐")

    print("="*70 + "\n")

    # 加载数据
    tensor_file = os.path.join(args.data_dir, f"{args.dataset}.npy")

    if not os.path.exists(tensor_file):
        print(f"错误：找不到文件 {tensor_file}")
        print(f"当前目录: {os.getcwd()}")
        print(f"请确保文件路径正确")
        return

    print(f"正在加载文件: {tensor_file}")
    data = np.load(tensor_file)
    print(f"✓ 加载成功")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    # 检测数据格式
    if len(data.shape) == 3:
        # 格式1: 3D张量
        print(f"\n✓ 检测到3D张量格式")
        num_source, num_dest, num_time = data.shape
        print(f"  源节点数: {num_source}")
        print(f"  目标节点数: {num_dest}")
        print(f"  时间点数: {num_time}")

        print(f"\n开始转换...")
        matrix = convert_3d_to_matrix(data, num_source, num_dest, num_time, args.time_axis)

    elif len(data.shape) == 2 and data.shape[1] == 4:
        # 格式2: 稀疏索引格式 (N, 4)
        print(f"\n✓ 检测到稀疏索引格式 (索引-值对)")
        print(f"  数据点数: {data.shape[0]}")
        print(f"  列数: {data.shape[1]} [源节点, 目标节点, 时间, 流量值]")

        # 提取信息
        sources = data[:, 0].astype(int)
        dests = data[:, 1].astype(int)
        times = data[:, 2].astype(int)
        values = data[:, 3]

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

        # 验证
        expected_points = num_source * num_dest * num_time
        actual_points = len(sources)

        print(f"\n数据验证:")
        print(f"  预期数据点数: {expected_points}")
        print(f"  实际数据点数: {actual_points}")
        print(f"  稀疏度: {(1 - actual_points/expected_points)*100:.2f}%")

        # 显示数据范围
        print(f"\n数据范围:")
        print(f"  源节点索引: [{sources.min()}, {sources.max()}]")
        print(f"  目标节点索引: [{dests.min()}, {dests.max()}]")
        print(f"  时间索引: [{times.min()}, {times.max()}]")
        print(f"  流量值: [{values.min():.4f}, {values.max():.4f}]")

        print(f"\n开始转换...")
        matrix = convert_sparse_to_matrix(data, num_source, num_dest, num_time, args.time_axis)

    else:
        print(f"\n✗ 错误：不支持的数据格式")
        print(f"  形状: {data.shape}")
        print(f"  期望格式:")
        print(f"    1. 3D张量: (源节点, 目标节点, 时间)")
        print(f"    2. 稀疏格式: (数据点数, 4)")
        return

    print(f"\n转换完成！")
    print(f"输出矩阵形状: {matrix.shape}")

    if args.time_axis == 'row':
        print(f"  - 行数（时间点）: {matrix.shape[0]}")
        print(f"  - 列数（链路数）: {matrix.shape[1]}")
        print(f"\n每一行代表一个时间点的网络状态")
        print(f"每一列代表一条网络链路（源节点->目标节点）")
    else:
        print(f"  - 行数（链路数）: {matrix.shape[0]}")
        print(f"  - 列数（时间点）: {matrix.shape[1]}")
        print(f"\n每一行代表一条网络链路（源节点->目标节点）")
        print(f"每一列代表一个时间点的网络状态 ⭐")

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
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = '_col_time' if args.time_axis == 'col' else '_row_time'
    output_path = os.path.join(args.output_dir, f'{args.dataset}_matrix{suffix}.npy')

    np.save(output_path, matrix)

    print(f"\n输出文件已保存到: {output_path}")

    # 验证保存的文件
    verify_matrix = np.load(output_path)
    if np.array_equal(matrix, verify_matrix):
        print("✓ 验证成功：保存的矩阵数据一致")
    else:
        print("✗ 警告：保存的矩阵数据不一致")

    print("="*70)

    # 显示示例数据
    if args.time_axis == 'col':
        print(f"\n示例数据（前5行链路，前10列时间点）:")
        print(f"{'':^12}", end='')
        for j in range(min(10, matrix.shape[1])):
            print(f"{f'时间{j}':^10}", end='')
        print()
        for i in range(min(5, matrix.shape[0])):
            link_src = i // num_dest
            link_dst = i % num_dest
            print(f"链路{i:3d}({link_src}->{link_dst})", end=' ')
            for j in range(min(10, matrix.shape[1])):
                print(f"{matrix[i, j]:^10.2f}", end='')
            print()
    else:
        print(f"\n示例数据（前5行时间点，前10列链路）:")
        print(f"{'':^8}", end='')
        for j in range(min(10, matrix.shape[1])):
            print(f"{f'链路{j}':^12}", end='')
        print()
        for i in range(min(5, matrix.shape[0])):
            print(f"时间{i:^5}", end=' ')
            for j in range(min(10, matrix.shape[1])):
                print(f"{matrix[i, j]:^12.2f}", end='')
            print()


if __name__ == "__main__":
    main()
