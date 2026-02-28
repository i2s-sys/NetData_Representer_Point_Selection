"""
将 Abilene.npy 网络流量张量转换为二维矩阵（每一列表示一个时间点）
输出形状: (3000, 144) - (时间点, 源节点×目标节点)

Abilene 数据集说明:
- 第一维 (12): 源节点 (source nodes)
- 第二维 (12): 目标节点 (destination nodes)
- 第三维 (3000): 时间点 (time points)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handleData.tensor_converter import TensorConverter
import numpy as np

def main():
    print("="*70)
    print("转换 Abilene 网络流量张量为二维矩阵（每列=一个时间点）")
    print("="*70 + "\n")

    # 创建转换器
    converter = TensorConverter('Abilene')

    print(f"原始张量形状: {converter.tensor.shape}")
    print(f"  - 源节点数量: {converter.tensor.shape[0]}")
    print(f"  - 目标节点数量: {converter.tensor.shape[1]}")
    print(f"  - 时间点数量: {converter.tensor.shape[2]}")
    print(f"  - 总链路数: {converter.tensor.shape[0] * converter.tensor.shape[1]} (源节点×目标节点)")
    print()

    # 转换为矩阵（每列=一个时间点）
    print("正在转换...")
    matrix = converter.convert_to_matrix(mode='time_user_item', save=True)

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
    print(f"  非零值比例: {(matrix != 0).sum() / matrix.size:.2%}")

    print(f"\n输出文件已保存到: ./handleData/output/Abilene_matrix_time_user_item.npy")
    print("="*70)

if __name__ == "__main__":
    main()
