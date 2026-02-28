# CostCO 在线学习预测模块

## 📁 文件说明

```
CostCO/
├── online_learning.py        # ⭐ 主要在线学习脚本
└── README.md              # 本文件
```

## 🎯 功能说明

本模块实现基于 Costco 模型的在线学习预测，用于网络流量预测。

### 核心流程

```
1. 【初始训练】
   使用前 N 列（如2000列）作为历史数据
   训练初始 Costco 模型
         ↓
2. 【预测】
   预测第 N+1 列
         ↓
3. 【增量更新】
   加入第 N+1 列的真实数据
   使用滑动窗口继续训练模型
         ↓
4. 【重复预测】
   预测第 N+2 列
         ↓
5. 【循环】
   重复步骤3-4，直到预测完所有数据
```

## 🚀 快速开始

### 步骤1：转换数据为矩阵

```bash
cd handleData
python convert_robust.py --dataset Geant_23_23_3000 --time_axis col
```

**输出：** `handleData/output/Geant_23_23_3000_matrix.npy` (529, 3000)

### 步骤2：运行在线学习预测

```bash
cd CostCO
python online_learning.py --dataset Geant_23_23_3000
```

**输出：**
- `output/predictions.npy` - 预测结果 (529, 999)
- `output/true_values.npy` - 真实值 (529, 999)

## ⚙️ 参数说明

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | Geant_23_23_3000 | 数据集名称 |
| `--matrix_dir` | ./handleData/output | 矩阵文件目录 |
| `--output_dir` | ./output | 预测结果输出目录 |
| `--initial_window` | 2000 | 初始训练窗口大小 |
| `--window_size` | 100 | 滑动窗口大小 |
| `--hidden_size` | 128 | LSTM隐藏层大小 |
| `--lr` | 1e-3 | 学习率 |
| `--epochs` | 5 | 每次更新的训练轮数 |

### 参数影响

- **initial_window**: 越大，初始模型训练越充分，但耗时越长
- **window_size**: 滑动窗口大小，影响模型对近期数据的敏感度
- **hidden_size**: LSTM隐藏层大小，越大越强但容易过拟合
- **lr**: 学习率，太大可能导致不收敛，太小收敛慢
- **epochs**: 每次更新的训练轮数，影响增量更新速度

### 参数调优示例

```bash
# 快速测试
python online_learning.py \
    --dataset Geant_23_23_3000 \
    --initial_window 500 \
    --window_size 50 \
    --epochs 3

# 标准配置（推荐）⭐
python online_learning.py \
    --dataset Geant_23_23_3000 \
    --initial_window 2000 \
    --window_size 100 \
    --epochs 5

# 高质量配置
python online_learning.py \
    --dataset Geant_23_23_3000 \
    --initial_window 2000 \
    --window_size 150 \
    --epochs 10 \
    --lr 1e-4
```

## 📊 模型架构

### Costco 模型结构

```python
CostcoModel(
    input_size=529,      # 链路数作为输入特征
    hidden_size=128,      # LSTM隐藏层大小
    num_layers=2,        # LSTM层数
    dropout=0.2          # Dropout比例
)
```

**组件：**
- **LSTM编码器**: 2层双向LSTM
- **全连接层**: 2层MLP，带ReLU和Dropout

### 输入输出

**输入：**
- 形状: `(1, window_size, num_links)`
  - batch_size: 1
  - seq_len: window_size (滑动窗口）
  - input_size: num_links (链路数）

**输出：**
- 形状: `(num_links, 1)`
  - 所有链路的预测值

## 💻 使用示例

### 基本使用

```bash
cd CostCO
python online_learning.py --dataset Geant_23_23_3000
```

### 使用不同的数据集

```bash
# Abilene 数据集
python online_learning.py --dataset Abilene_12_12_3000

# Geant 数据集
python online_learning.py --dataset Geant_23_23_3000
```

### 自定义参数

```bash
python online_learning.py \
    --dataset Geant_23_23_3000 \
    --initial_window 2000 \
    --window_size 100 \
    --hidden_size 128 \
    --lr 1e-3 \
    --epochs 5
```

## 📈 预测结果分析

### 加载预测结果

```python
import numpy as np

# 加载预测和真实值
predictions = np.load('output/predictions.npy')
true_values = np.load('output/true_values.npy')

print(f"预测结果形状: {predictions.shape}")  # (529, 999)
print(f"真实值形状: {true_values.shape}")      # (529, 999)
```

### 分析特定链路

```python
# 查看链路0的预测
link_id = 0
link_pred = predictions[link_id, :]
link_true = true_values[link_id, :]

print(f"链路0的MAE: {np.abs(link_pred - link_true).mean():.4f}")
```

### 可视化预测结果

```python
import matplotlib.pyplot as plt

# 选择前10条链路进行可视化
n_links_to_plot = 10
time_steps = range(len(true_values[0]))

plt.figure(figsize=(15, 4 * n_links_to_plot))

for i in range(n_links_to_plot):
    plt.subplot(n_links_to_plot, 1, i+1)
    plt.plot(time_steps[:200], true_values[i, :200], 
             label='真实值', alpha=0.7, linewidth=1)
    plt.plot(time_steps[:200], predictions[i, :200], 
             label='预测值', alpha=0.7, linewidth=1)
    plt.title(f'链路{i}的预测')
    plt.xlabel('预测步骤')
    plt.ylabel('流量值')
    plt.legend()

plt.tight_layout()
plt.show()
```

### 计算评估指标

```python
# 整体指标
mae = np.mean(np.abs(predictions - true_values))
rmse = np.sqrt(np.mean((predictions - true_values) ** 2))

print(f"整体指标:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")

# 每条链路的指标
mae_per_link = np.abs(predictions - true_values).mean(axis=1)
print(f"\n各链路的MAE (前10条):")
for i in range(10):
    print(f"  链路{i:3d}: {mae_per_link[i]:.4f}")
```

## 🔍 数据流程

### 输入数据

```
handleData/output/Geant_23_23_3000_matrix.npy
形状: (529, 3000)
       ↓      ↓
    链路    时间点

每行 = 一条网络链路（源节点->目标节点）
每列 = 一个时间点
```

### 训练数据

```
初始训练:
  X: (2000, 529)  - 前2000列
  y: (529,)          - 第2001列

增量更新:
  X: (100, 529)    - 滑动窗口
  y: (529,)          - 当前列
```

### 输出数据

```
output/predictions.npy
形状: (529, 999)
       ↓      ↓
    链路    预测次数

每行 = 一条链路的预测序列
每列 = 一次预测（针对一个时间点）
```

## 📋 项目结构

```
NetData/
├── CostCO/                        # Costco 在线学习模块 ⭐
│   ├── online_learning.py           # 主要预测脚本
│   └── README.md                 # 本文件
├── handleData/                    # 数据转换模块
│   ├── convert_robust.py          # 数据转换脚本
│   ├── convert_batch.py            # 批量转换
│   └── output/                   # 转换后的矩阵
│       ├── Geant_23_23_3000_matrix.npy
│       └── Abilene_12_12_3000_matrix.npy
├── data/                         # 原始数据
│   ├── Geant_23_23_3000.npy
│   └── Abilene_12_12_3000.npy
├── Amain.py                     # 主训练脚本
├── Amodels.py                   # 模型定义
└── config.py                    # 配置文件
```

## 🎓 在线学习原理

### 为什么使用在线学习？

1. **数据漂移**: 网络流量模式会随时间变化
2. **实时性**: 需要及时适应新数据
3. **增量更新**: 不需要重新训练整个模型

### 与离线学习的区别

| 特性 | 离线学习 | 在线学习（本实现）|
|------|---------|----------------|
| 训练数据 | 全部历史数据 | 滑动窗口 |
| 更新方式 | 重新训练 | 增量更新 |
| 计算成本 | 高 | 低 |
| 适应性 | 差 | 好 |

## 🐛 常见问题

### Q: 内存不足？

A: 尝试调整参数：
```bash
python online_learning.py \
    --dataset Geant_23_23_3000 \
    --window_size 50 \
    --hidden_size 64
```

### Q: 预测误差大？

A: 尝试：
- 增加 `--epochs` (如10或15)
- 调整 `--lr` (如1e-4)
- 增大 `--window_size` (如150)

### Q: 训练太慢？

A: 尝试：
- 减小 `--initial_window` (如1000)
- 减小 `--epochs` (如3)
- 减小 `--window_size` (如50)

### Q: 如何保存中间模型？

A: 在代码中添加保存逻辑（约第180行）:
```python
if pred_col % 100 == 0:
    torch.save(self.model.state_dict(), 
               f'model_checkpoint_col_{pred_col}.pt')
```

## ✅ 完整执行流程

```bash
# 1. 转换 Geant 数据为矩阵
cd handleData
python convert_robust.py --dataset Geant_23_23_3000 --time_axis col

# 2. 运行在线学习预测
cd ../CostCO
python online_learning.py --dataset Geant_23_23_3000

# 3. 查看预测结果
python -c "import numpy as np; pred = np.load('output/predictions.npy'); print(f'预测结果形状: {pred.shape}')"
```

## 📖 相关文档

- **`../handleData/README.md`** - 数据转换说明
- **`../handleData/USAGE_GUIDE.md`** - 转换脚本使用指南
- **`../Amain.py`** - 主训练流程参考

---

**快速开始：**

```bash
cd CostCO
python online_learning.py --dataset Geant_23_23_3000
```

**输出将保存在：** `output/predictions.npy` 和 `output/true_values.npy`
