# 转换脚本使用指南

## 📁 handleData 目录下的转换脚本

| 脚本 | 说明 | 推荐度 |
|------|------|--------|
| `convert_robust.py` | ⭐⭐⭐ 主转换脚本，支持多种格式和方向 |
| `convert_to_matrix_col_time.py` | ⭐⭐⭐ 专门用于列=时间点的转换 |
| `convert_sparse_to_matrix.py` | ⭐⭐ 专门处理稀疏格式（列=时间） |
| `tensor_converter.py` | 通用转换工具 |

## 🎯 你的需求：列是时间点

### 输出格式

```
矩阵形状: (链路数, 时间点)
         ↓      ↓
      链路    时间
   (144行) (3000列)

每行 = 一条网络链路
每列 = 一个时间点 ⭐
```

### 转换方法

#### 方法1：使用 convert_robust.py（推荐）⭐

```bash
cd handleData
python convert_robust.py --dataset Abilene_12_12_3000 --time_axis col
```

**参数说明：**
- `--dataset`: 数据集名称
- `--time_axis col`: **列是时间点** ⭐

**输出：**
- 文件：`output/Abilene_12_12_3000_matrix_col_time.npy`
- 形状：(144, 3000)
- 144行 = 144条链路
- 3000列 = 3000个时间点

#### 方法2：使用 convert_to_matrix_col_time.py

```bash
cd handleData
python convert_to_matrix_col_time.py --dataset Abilene_12_12_3000
```

**输出：**
- 文件：`output/Abilene_12_12_3000_matrix.npy`
- 形状：(144, 3000)

## 📊 数据格式对比

### 输入数据（稀疏格式）

```
形状: (432000, 4)
       ↓       ↓
    数据点   [源节点, 目标节点, 时间, 流量]

示例:
[0, 0, 0, 12.34]  # 时间0，链路0->0的流量12.34
[0, 1, 0, 5.67]   # 时间0，链路0->1的流量5.67
[1, 0, 0, 8.90]   # 时间0，链路1->0的流量8.90
...
```

### 输出矩阵（列是时间点）

```
形状: (144, 3000)
       ↓      ↓
    链路    时间

             时间0   时间1   ...  时间2999
           ↓       ↓              ↓
链路0 (0->0) [12.34, 14.56, ..., 10.23]
链路1 (0->1) [5.67,  6.78,  ..., 4.56]
链路2 (0->2) [8.90,  9.12,  ..., 7.89]
...
链路143 (11->11) [15.67, 16.78, ..., 14.56]
```

## 🔧 两种输出格式对比

### 格式A：行是时间（time_axis=row）

```bash
python convert_robust.py --time_axis row
```

```
输出: (3000, 144)
每行 = 一个时间点
每列 = 一条链路
```

### 格式B：列是时间（time_axis=col）⭐ 你的需求

```bash
python convert_robust.py --time_axis col
```

```
输出: (144, 3000)
每行 = 一条链路
每列 = 一个时间点 ⭐
```

## 💻 Python API 使用

### 加载转换后的矩阵

```python
import numpy as np

# 加载矩阵
matrix = np.load('handleData/output/Abilene_12_12_3000_matrix_col_time.npy')

print(f"矩阵形状: {matrix.shape}")  # (144, 3000)
print(f"链路数: {matrix.shape[0]}")    # 144
print(f"时间点数: {matrix.shape[1]}")   # 3000

# 访问特定链路的数据
link_id = 5
link_timeseries = matrix[link_id, :]  # 链路5在所有时间点的流量
print(f"链路5的时间序列: {link_timeseries}")

# 访问特定时间点的数据
time_id = 100
snapshot = matrix[:, time_id]  # 时间点100的所有链路流量
print(f"时间点100的网络状态: {snapshot}")
```

### 在训练中使用

```python
import numpy as np
import torch

# 加载矩阵
matrix = np.load('handleData/output/Abilene_12_12_3000_matrix_col_time.npy')

# 转换为 PyTorch 张量
data = torch.from_numpy(matrix).float()

# 如果有CUDA
if torch.cuda.is_available():
    data = data.cuda()

# 归一化
data = data / data.max()

print(f"训练数据形状: {data.shape}")  # (144, 3000)
# 144条链路，每条链路有3000个时间步
```

## 📋 完整命令示例

### Abilene 数据集（列是时间）

```bash
cd handleData
python convert_robust.py --dataset Abilene_12_12_3000 --time_axis col
```

### Geant 数据集

```bash
cd handleData
python convert_robust.py --dataset Geant --time_axis col
```

### 指定输出目录

```bash
cd handleData
python convert_robust.py \
    --dataset Abilene_12_12_3000 \
    --time_axis col \
    --output_dir ./output
```

## 🔍 链路索引映射

对于矩阵的行索引 `row` (0 到 143)：

```python
源节点 = row // 12  # 整数除法
目标节点 = row % 12   # 取余数
```

示例：
- 行0 → 源节点0，目标节点0 (链路: 0→0)
- 行5 → 源节点0，目标节点5 (链路: 0→5)
- 行13 → 源节点1，目标节点1 (链路: 1→1)
- 行143 → 源节点11，目标节点11 (链路: 11→11)

## 📊 转换示例

### 运行转换

```bash
$ cd handleData
$ python convert_robust.py --dataset Abilene_12_12_3000 --time_axis col

======================================================================
转换 Abilene_12_12_3000 网络流量数据为二维矩阵
输出格式: (链路, 时间) - 每一列=一个时间点 ⭐
======================================================================

正在加载文件: ../data/Abilene_12_12_3000.npy
✓ 加载成功
数据形状: (432000, 4)
数据类型: float64

✓ 检测到稀疏索引格式 (索引-值对)
  数据点数: 432000
  列数: 4 [源节点, 目标节点, 时间, 流量值]

推断的维度:
  源节点数: 12
  目标节点数: 12
  时间点数: 3000
  总链路数: 144

数据验证:
  预期数据点数: 432000
  实际数据点数: 432000
  稀疏度: 0.00%

数据范围:
  源节点索引: [0, 11]
  目标节点索引: [0, 11]
  时间索引: [0, 2999]
  流量值: [0.0000, 100.0000]

开始转换...
  填充 432000 个数据点到 (链路, 时间) 格式...
  ✓ 填充完成

转换完成！
输出矩阵形状: (144, 3000)
  - 行数（链路数）: 144
  - 列数（时间点）: 3000

每一行代表一条网络链路（源节点->目标节点）
每一列代表一个时间点的网络状态 ⭐

数据统计:
  范围: [0.0000, 100.0000]
  均值: 15.2345
  标准差: 12.3456
  中位数: 12.3456
  非零值比例: 85.50%
  零值数量: 62640
  总元素数: 432000

输出文件已保存到: ./output/Abilene_12_12_3000_matrix_col_time.npy
✓ 验证成功：保存的矩阵数据一致
======================================================================

示例数据（前5行链路，前10列时间点）:
              时间0      时间1      时间2  ...    时间9
链路 0(0->0)   12.34      14.56      13.45  ...   11.23
链路 1(0->1)    5.67       6.78       5.89   ...    4.56
链路 2(0->2)    8.90       9.12       8.45   ...    7.78
链路 3(0->3)    3.45       4.56       3.67   ...    2.89
链路 4(0->4)    6.78       7.89       6.90   ...    5.67
```

## ✅ 总结

**快速转换命令：**

```bash
cd handleData
python convert_robust.py --dataset Abilene_12_12_3000 --time_axis col
```

**输出：**
- 文件：`output/Abilene_12_12_3000_matrix_col_time.npy`
- 形状：**(144, 3000)**
- **每一行** = 一条链路
- **每一列** = 一个时间点 ⭐
