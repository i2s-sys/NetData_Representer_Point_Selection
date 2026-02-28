# 网络流量数据结构说明

## 数据集：Abilene

Abilene 是一个真实的互联网骨干网络流量数据集，用于网络流量预测和异常检测研究。

## 数据维度

### 3D张量结构：`Abilene.npy`

```
形状: (12, 12, 3000)
       ↓   ↓    ↓
    源节点 目标节点  时间点
```

**维度说明：**

1. **第一维 (12)**: 源节点 (Source Nodes)
   - 表示网络中的12个路由器/网络节点
   - 作为流量的起始点

2. **第二维 (12)**: 目标节点 (Destination Nodes)
   - 表示同样的12个网络节点
   - 作为流量的目的地

3. **第三维 (3000)**: 时间点 (Time Points)
   - 表示连续的3000个时间步
   - 每个时间点代表一个网络状态快照

## 网络链路

总链路数 = 源节点数 × 目标节点数 = 12 × 12 = **144条链路**

每条链路：`源节点 -> 目标节点` 的网络流量

### 链路索引映射

对于张量 `tensor[i, j, k]`：

- `i`: 源节点索引 (0-11)
- `j`: 目标节点索引 (0-11)
- `k`: 时间点索引 (0-2999)
- `tensor[i, j, k]`: 在时间k，从节点i到节点j的网络流量值

## 转换为二维矩阵

### 模式1: 每列=一条链路（推荐用于时间序列分析）

```python
# (source, dest, time) -> (time, source*dest)
# 输出形状: (3000, 144)

num_source, num_dest, num_time = tensor.shape
matrix = tensor.transpose(2, 0, 1).reshape(num_time, num_source * num_dest)
```

**矩阵结构：**
```
         列0  列1  ...  列143
         ↓    ↓         ↓
       链路0 链路1 ... 链路143
行0  [  v1   v2  ...  v144 ]  ← 时间点0
行1  [  v1   v2  ...  v144 ]  ← 时间点1
行2  [  v1   v2  ...  v144 ]  ← 时间点2
...
行2999 [  v1   v2  ...  v144 ]  ← 时间点2999
```

- **每一行** = 一个时间点的网络快照（所有144条链路的流量）
- **每一列** = 一条特定链路的时间序列（3000个时间点的流量）

**列索引映射到链路：**
- 列 `c` 对应链路：`源节点 = c // 12`, `目标节点 = c % 12`
- 例如：列5 → 源节点0，目标节点5
- 例如：列13 → 源节点1，目标节点1
- 例如：列143 → 源节点11，目标节点11

### 模式2: 每行=一条链路（推荐用于链路特征分析）

```python
# (source, dest, time) -> (source*dest, time)
# 输出形状: (144, 3000)

num_source, num_dest, num_time = tensor.shape
matrix = tensor.reshape(num_source * num_dest, num_time)
```

**矩阵结构：**
```
         列0  列1  ...  列2999
         ↓    ↓         ↓
       时间0 时间1 ... 时间2999
行0   [  v1   v2  ...  v3000 ]  ← 链路0 (0->0)
行1   [  v1   v2  ...  v3000 ]  ← 链路1 (0->1)
行2   [  v1   v2  ...  v3000 ]  ← 链路2 (0->2)
...
行143  [  v1   v2  ...  v3000 ]  ← 链路143 (11->11)
```

- **每一行** = 一条链路的时间序列
- **每一列** = 一个时间点所有链路的流量

### 模式3: 按源节点展开

```python
# (source, dest, time) -> (source, dest*time)
# 输出形状: (12, 36000)

matrix = tensor.reshape(num_source, num_dest * num_time)
```

- 每一行代表一个源节点
- 包含该节点到所有目标节点在所有时间点的流量

### 模式4: 按目标节点展开

```python
# (source, dest, time) -> (dest, source*time)
# 输出形状: (12, 36000)

matrix = tensor.transpose(1, 0, 2).reshape(num_dest, num_source * num_time)
```

- 每一行代表一个目标节点
- 包含所有源节点到该目标节点在所有时间点的流量

## 网络图表示

```
Abilene 网络拓扑（12个节点）:

    N0 ───────── N1 ───────── N2
     │            │            │
     │            │            │
    N3 ───────── N4 ───────── N5
     │            │            │
     │            │            │
    N6 ───────── N7 ───────── N8
     │            │            │
     │            │            │
    N9 ───────── N10 ──────── N11

每条连线可能有双向流量：
- N0 → N1: tensor[0, 1, :]
- N1 → N0: tensor[1, 0, :]
```

## 实际应用场景

### 1. 时间序列预测
```python
# 使用模式1：每列=一条链路
matrix = tensor.transpose(2, 0, 1).reshape(3000, 144)

# 预测：已知前T-1个时间点，预测第T个时间点
# 模型输入: matrix[:T-1, :]  # 历史流量
# 模型输出: matrix[T-1, :]   # 当前流量
```

### 2. 链路异常检测
```python
# 使用模式2：每行=一条链路
matrix = tensor.reshape(144, 3000)

# 对每条链路的时间序列进行异常检测
for link_id in range(144):
    link_timeseries = matrix[link_id, :]
    # 检测异常...
```

### 3. 网络状态快照分析
```python
# 使用模式1：每列=一条链路
matrix = tensor.transpose(2, 0, 1).reshape(3000, 144)

# 分析每个时间点的网络状态
for time_id in range(3000):
    snapshot = matrix[time_id, :]  # 144条链路的流量
    # 分析网络拥堵情况...
```

### 4. 源节点流量分析
```python
# 使用模式3：按源节点展开
matrix = tensor.reshape(12, 36000)

# 分析每个源节点的流量模式
for src_id in range(12):
    src_traffic = matrix[src_id, :]  # 源节点src_id的所有流量
    # 分析流量模式...
```

## 数据统计信息

典型特征：
- 稀疏性：许多链路在某些时间点流量为0
- 周期性：工作日/周末模式
- 自相关性：相邻时间点流量高度相关
- 突发性：偶发的流量突发事件

## 与用户-物品评分数据的区别

| 特征 | 网络流量数据 | 用户-物品评分数据 |
|------|------------|----------------|
| 第一维 | 源节点 (12) | 用户 (数千) |
| 第二维 | 目标节点 (12) | 物品 (数千) |
| 第三维 | 时间 (3000) | 通常无或很少 |
| 数据类型 | 连续值 (流量) | 离散值 (评分) |
| 稀疏性 | 中等 | 高 |
| 对称性 | 通常不对称 (i≠j时) | 不对称 |

## 代码示例

### 计算某条链路的流量统计

```python
import numpy as np

tensor = np.load('./data/Abilene.npy')

# 计算链路 节点5->节点3 的统计
link_5_3 = tensor[5, 3, :]

print(f"链路 5->3 流量统计:")
print(f"  平均值: {link_5_3.mean():.2f}")
print(f"  最大值: {link_5_3.max():.2f}")
print(f"  最小值: {link_5_3.min():.2f}")
print(f"  标准差: {link_5_3.std():.2f}")
print(f"  零流量比例: {(link_5_3 == 0).sum() / len(link_5_3):.2%}")
```

### 查找某时间点最繁忙的链路

```python
import numpy as np

tensor = np.load('./data/Abilene.npy')

# 找时间点1000最繁忙的链路
time_id = 1000
snapshot = tensor[:, :, time_id]

# 找最大值及其位置
max_traffic = snapshot.max()
src, dest = np.unravel_index(snapshot.argmax(), snapshot.shape)

print(f"时间点 {time_id} 最繁忙的链路: {src} -> {dest}")
print(f"流量: {max_traffic:.2f}")
```

### 可视化网络流量矩阵

```python
import numpy as np
import matplotlib.pyplot as plt

tensor = np.load('./data/Abilene.npy')

# 显示时间点1000的流量矩阵（热力图）
time_id = 1000
traffic_matrix = tensor[:, :, time_id]

plt.figure(figsize=(10, 8))
plt.imshow(traffic_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Traffic')
plt.title(f'Network Traffic at Time {time_id}')
plt.xlabel('Destination Node')
plt.ylabel('Source Node')
plt.show()
```

## 总结

- **数据本质**：网络流量数据，不是用户-物品数据
- **第一维**：源节点（流量起始点）
- **第二维**：目标节点（流量目的地）
- **第三维**：时间（网络状态快照）
- **链路数**：源节点 × 目标节点 = 144条
- **转换模式选择**：根据分析需求选择合适的转换模式
