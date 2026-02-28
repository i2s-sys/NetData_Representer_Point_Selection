"""
将索引-值对格式转换为矩阵

输入格式: (N, 4) - 每行是 [源节点, 目标节点, 时间, 流量值]
输出格式: (时间, 源节点×目标节点) - 每列是一条链路

使用方法:
    cd handleData
    python convert_sparse_to_matrix.py --dataset Abilene_12_12_3000
"""

import numpy as np
import argparse
import os


def convert_sparse_to_matrix(dataset_name, data_dir='../data', output_dir='./output'):
    """
    将索引-值对格式转换为矩阵

    输入: (N, 4) - [source, dest, time, value]
    输出: (time, source*dest) - 每列是一条链路
    """
    print("="*70)
    print(f"转换 {dataset_name} 稀疏格式为矩阵")
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

    # 检查数据格式
    if sparse_data.shape[1] != 4:
        print(f"\n✗ 错误：期望4列 [源节点, 目标节点, 时间, 值]")
        print(f"  实际列数: {sparse_data.shape[1]}")
        return None

    print(f"  数据类型: {sparse_data.dtype}")
    print(f"  总数据点: {sparse_data.shape[0]}")

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

    print(f"\n推断的维度:")
    print(f"  源节点数: {num_source}")
    print(f"  目标节点数: {num_dest}")
    print(f"  时间点数: {num_time}")
    print(f"  总链路数: {num_source * num_dest}")

    # 验证维度是否合理
    expected_points = num_source * num_dest * num_time
    actual_points = len(sparse_data)

    print(f"\n验证:")
    print(f"  预期数据点数: {expected_points}")
    print(f"  实际数据点数: {actual_points}")
    print(f"  稀疏度: {(1 - actual_points/expected_points)*100:.2f}%")

    # 转换为矩阵格式
    print(f"\n开始转换...")
    print(f"目标形状: (时间, 源节点×目标节点) = ({num_time}, {num_source * num_dest})")

    # 方法1: 直接重塑（如果数据是有序的）
    # 假设数据按 (source, dest, time) 排序
    try:
        # 尝试直接reshape为3D张量
        tensor_3d = values.reshape(num_source, num_dest, num_time)
        print(f"✓ 直接reshape成功，数据是有序的")

        # 转换为矩阵: (source, dest, time) -> (time, source*dest)
        matrix = tensor_3d.transpose(2, 0, 1).reshape(num_time, num_source * num_dest)

    except Exception as e:
        print(f"✗ 直接reshape失败: {e}")
        print(f"  数据可能不是有序的，使用构造方法...")

        # 方法2: 构造稀疏矩阵
        # 初始化矩阵
        matrix = np.zeros((num_time, num_source * num_dest), dtype=values.dtype)

        # 填充矩阵
        print(f"  填充 {len(sources)} 个数据点...")
        for i, (s, d, t, v) in enumerate(zip(sources, dests, times, values)):
            col = s * num_dest + d  # 计算列索引
            matrix[t, col] = v

            if (i + 1) % 100000 == 0:
                print(f"    进度: {i+1}/{len(sources)}")

        print(f"✓ 填充完成")

    print(f"\n转换完成！")
    print(f"  输出形状: {matrix.shape}")
    print(f"  行数（时间点）: {matrix.shape[0]}")
    print(f"  列数（链路数）: {matrix.shape[1]}")
    print(f"  每一列代表一条网络链路（源节点->目标节点）")
    print(f"  每一行代表一个时间点的网络状态")

    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print(f"  均值: {matrix.mean():.4f}")
    print(f"  标准差: {matrix.std():.4f}")
    print(f"  中位数: {np.median(matrix):.4f}")
    print(f"  非零值比例: {(matrix != 0).sum() / matrix.size:.2%}")
    print(f"  零值数量: {(matrix == 0).sum()}")

    # 保存矩阵
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}_matrix_time_link.npy')

    np.save(output_path, matrix)
    print(f"\n✓ 已保存到: {output_path}")

    # 验证
    verify = np.load(output_path)
    if np.array_equal(matrix, verify):
        print(f"✓ 验证成功：数据一致")
    else:
        print(f"✗ 警告：数据不一致")

    print("="*70)

    return matrix


def main():
    parser = argparse.ArgumentParser(description='将稀疏格式转换为矩阵')
    parser.add_argument('--dataset', type=str, default='Abilene_12_12_3000',
                        help='数据集名称 (默认: Abilene_12_12_3000)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录 (默认: ../data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')

    args = parser.parse_args()

    convert_sparse_to_matrix(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
