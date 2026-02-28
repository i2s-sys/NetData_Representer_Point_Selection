# 快速开始指南

## 🚀 30秒快速转换

### 1. 进入 handleData 目录

```bash
cd handleData
```

### 2. 运行转换脚本

```bash
python convert_network_tensor.py
```

### 3. 完成！

转换后的矩阵保存在：`output/Abilene_matrix_time_link.npy`

---

## 📋 可用脚本

所有脚本都在 `handleData/` 目录下：

| 脚本 | 说明 | 推荐度 |
|------|------|--------|
| `convert_network_tensor.py` | 网络张量转换（通用） | ⭐⭐⭐⭐⭐ |
| `convert_robust.py` | 不依赖配置文件的转换 | ⭐⭐⭐⭐ |
| `tensor_converter.py` | 支持多种转换模式 | ⭐⭐⭐⭐⭐ |
| `convert_abilene.py` | Abilene专用转换 | ⭐⭐⭐ |

## 🎯 常用命令

### 转换 Abilene 数据集

```bash
cd handleData
python convert_network_tensor.py --dataset Abilene
```

### 转换 Geant 数据集

```bash
cd handleData
python convert_network_tensor.py --dataset Geant
```

### 使用不依赖配置文件的版本

```bash
cd handleData
python convert_robust.py
```

### 使用 tensor_converter（支持多种模式）

```python
cd handleData
python -c "from tensor_converter import TensorConverter; converter = TensorConverter('Abilene'); converter.convert_all_modes()"
```

## 📁 文件结构

```
NetData/
├── handleData/              # 数据转换工具目录
│   ├── convert_network_tensor.py   # 主要转换脚本 ⭐
│   ├── convert_robust.py         # 不依赖配置文件 ⭐
│   ├── tensor_converter.py       # 支持多种转换模式 ⭐
│   ├── convert_abilene.py       # Abilene专用
│   ├── example_usage.py         # 使用示例
│   ├── output/                  # 转换后的矩阵输出
│   ├── README.md               # 完整文档
│   ├── QUICKSTART.md           # 本文件
│   ├── NETWORK_DATA_EXPLANATION.md  # 网络数据说明
│   └── TROUBLESHOOTING.md     # 错误排查
├── data/                     # 原始数据
│   ├── Abilene.npy           # 原始张量
│   ├── Abilene.ini           # 配置文件
│   └── ...
├── Amain.py                 # 主训练脚本
├── Amodels.py              # 模型定义
└── config.py               # 数据加载配置
```

## 💻 Python 使用示例

### 基础使用

```python
from tensor_converter import TensorConverter

# 创建转换器
converter = TensorConverter('Abilene')

# 转换为矩阵（每列=一条链路）
matrix = converter.convert_to_matrix(mode='time_link', save=True)

print(f"转换完成！形状: {matrix.shape}")  # (3000, 144)
```

### 加载已转换的矩阵

```python
import numpy as np

# 加载矩阵
matrix = np.load('handleData/output/Abilene_matrix_time_link.npy')

print(f"矩阵形状: {matrix.shape}")  # (3000, 144)
print(f"数据范围: [{matrix.min():.2f}, {matrix.max():.2f}]")
```

### 在训练中使用

```python
import numpy as np
import torch

# 加载矩阵
matrix = np.load('handleData/output/Abilene_matrix_time_link.npy')

# 转换为 PyTorch 张量
data = torch.from_numpy(matrix).float()

# 如果有CUDA
if torch.cuda.is_available():
    data = data.cuda()

# 归一化
data = data / data.max()

# 现在可以用于训练
print(f"训练数据形状: {data.shape}")  # (3000, 144)
```

## 🔄 转换模式对比

### time_link 模式（推荐）

```python
# 输入: (12, 12, 3000)
# 输出: (3000, 144)

from tensor_converter import TensorConverter
converter = TensorConverter('Abilene')
matrix = converter.convert_to_matrix(mode='time_link', save=True)
```

- **每行**: 一个时间点的网络快照
- **每列**: 一条网络链路的时间序列
- **适用**: 时间序列预测、网络状态分析

### flatten 模式

```python
# 输入: (12, 12, 3000)
# 输出: (144, 3000)

from tensor_converter import TensorConverter
converter = TensorConverter('Abilene')
matrix = converter.convert_to_matrix(mode='flatten', save=True)
```

- **每行**: 一条链路的时间序列
- **每列**: 一个时间点
- **适用**: 链路特征分析、聚类

### avg 模式

```python
# 输入: (12, 12, 3000)
# 输出: (12, 12)

from tensor_converter import TensorConverter
converter = TensorConverter('Abilene')
matrix = converter.convert_to_matrix(mode='avg', save=True)
```

- **每行**: 源节点
- **每列**: 目标节点
- **适用**: 静态网络分析、矩阵补全

## ⚡ 常见问题快速解决

### 问题1: 找不到文件

```bash
# 确保在 handleData 目录下
cd handleData
pwd  # 检查当前目录

# 确认文件存在
ls ../data/Abilene.npy
```

### 问题2: 配置文件错误

使用不依赖配置文件的版本：

```bash
cd handleData
python convert_robust.py
```

### 问题3: 导入错误

```bash
# 确保在 handleData 目录下运行
cd handleData
python convert_network_tensor.py
```

## 📊 输出示例

### Abilene 数据集转换结果

```
输出文件: handleData/output/Abilene_matrix_time_link.npy
矩阵形状: (3000, 144)

数据统计:
  - 范围: [0.00, 100.00]
  - 均值: 15.23
  - 非零值比例: 85.50%
```

### Geant 数据集转换结果

```
输出文件: handleData/output/Geant_matrix_time_link.npy
矩阵形状: (6000, 144)

数据统计:
  - 范围: [0.00, 200.00]
  - 均值: 25.67
  - 非零值比例: 78.30%
```

## 🎓 下一步

1. **查看完整文档**: `README.md`
2. **了解数据结构**: `NETWORK_DATA_EXPLANATION.md`
3. **查看使用示例**: `example_usage.py`
4. **开始训练**: 参考 `../Amain.py`

## 💡 提示

- ✅ 所有转换脚本都应在 `handleData` 目录下运行
- ✅ 转换后的矩阵保存在 `handleData/output/` 目录
- ✅ 可以多次运行，不会影响原始数据
- ✅ 大数据集转换可能需要一些时间

---

**需要帮助？** 查看 `TROUBLESHOOTING.md` 获取详细的错误排查指南。
