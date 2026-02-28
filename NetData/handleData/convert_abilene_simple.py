"""
将 Abilene.npy 网络流量张量转换为二维矩阵（每一列表示一个时间点）
输出形状: (3000, 144) - (时间点, 源节点×目标节点)

Abilene 数据集说明:
- 第一维 (12): 源节点 (source nodes)
- 第二维 (12): 目标节点 (destination nodes)
- 第三维 (3000): 时间点 (time points)
"""

import numpy as np
import configparser
import json
import os

def main():
    print("="*70)
    print("转换 Abilene 网络流量张量为二维矩阵（每列=一个时间点）")
    print("="*70 + "\n")

    # 加载配置
    config_file = '../../data/Abilene.ini'
    tensor_file = '../../data/Abilene.npy'

    # 读取配置
    conf = configparser.ConfigParser()
    conf.read(config_file)
    ndim = np.array(json.loads(conf.get("Data_Setting", "ndim")))

    print(f"配置信息:")
    print(f"  - 维度 ndim: {ndim}")
    print(f"  - 源节点数量: {ndim[0]}")
    print(f"  - 目标节点数量: {ndim[1]}")
    print(f"  - 时间点数量: {ndim[2]}")
    print(f"  - 总链路数: {ndim[0] * ndim[1]} (源节点×目标节点)")
    print()

    # 加载张量
    print(f"正在加载张量: {tensor_file}")
    tensor = np.load(tensor_file)
    print(f"原始张量形状: {tensor.shape}")
    print()

    # 转换为矩阵（每列=一个时间点）
    # (source, destination, time) -> (time, source * destination)
    print("正在转换...")
    num_source, num_dest, num_time = tensor.shape
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
    print(f"  非零值比例: {(matrix != 0).sum() / matrix.size:.2%}")

    # 保存矩阵
    output_dir = './handleData/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Abilene_matrix_time_user_item.npy')

    np.save(output_path, matrix)
    print(f"\n输出文件已保存到: {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
