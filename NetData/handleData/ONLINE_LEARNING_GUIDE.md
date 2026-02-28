# Costco 在线学习预测 - 完整指南

## 🎯 方案概述

### 在线学习流程

```
1. 【�】使用前2000列作为历史数据，训练初始模型
             ↓
2. 【预测】预测第2001列
             ↓
3. 【增量更新】加入第2001列数据，继续训练
             ↓
4. 【预测】预测第2002列
             ↓
5. 【重复3-4】...直到预测完所有数据
```

### 数据格式

```
矩阵: (529, 3000)
       ↓      ↓
    链路    时间点

训练窗口: 列 0-1999 (2000列)
预测范围: 列 2001-2999 (999次预测)
```

## 🚀 快速开始

### 步骤1：转换数据为矩阵

```bash
cd handleData
python convert_robust.py --dataset Geant_23_23_3000 --time_axis col
```

**输出：**
- 文件：`output/Geant_23_23_3000_matrix.npy`
- 形状：(529, 3000)

### 步骤2：运行在线学习预测

```bash
cd handleData
python online_predict_simple.py --dataset Geant_23_23_3000
```

**输出：**
- `output/predictions/predictions.npy` - 预测结果 (529, 999)
- `output/predictions/true_values.npy` - 真实值 (529, 999)

## 📊 详细流程说明

### 阶段1：初始训练

```
输入：列 0-1999 (2000个时间点）
      ↓
    Costco 模型
      ↓
模型学习：529条链路的历史模式
```

### 阶段2：在线预测循环

```
对于每个预测列 t = 2001 到 2999:

  ┌─────────────────────────────────┐
  │  预测阶段                      │
  └─────────────────────────────────┘
              ↓
  使用最近100列 (t-100 到 t-1)
              ↓
    Costco 模型
              ↓
      预测列 t
              ↓
  ┌─────────────────────────────────┐
  │  增量更新阶段              │
  └─────────────────────────────────┘
              ↓
  加入列 t 的真实数据
              ↓
  使用滑动窗口 (t-99 到 t)
              ↓
    Costco 模型
              ↓
    继续训练 (5 epochs)
              ↓
  模型更新完成，准备预测下一列
```

## ⚙️ 参数说明

### 在线预测器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--initial_window` | 2000 | 初始训练窗口大小（历史数据） |
| `--window_size` | 100 | 滑动窗口大小（用于预测） |
| `--hidden_size` | 128 | LSTM隐藏层大小 |
| `--lr` | 1e-3 | 学习率 |
| `--epochs` | 5 | 每次更新的训练轮数 |

### 参数调优建议

```bash
# 快速训练（适合调试）
python online_predict_simple.py \
    --dataset Geant_23_23_3000 \
    --initial_window 500 \
    --window_size 50 \
    --epochs 3

# 标准训练（推荐）
python online_predict_simple.py \
    --dataset Geant_23_23_3000 \
    --initial_window 2000 \
    --window_size 100 \
    --epochs 5

# 高质量训练（需要更多时间）
python online_predict_simple.py \
    --dataset Geant_23_23_3000 \
    --initial_window 2000 \
    --window_size 150 \
    --epochs 10 \
    --lr 1e-4
```

## 📖 代码实现

### Costco 模型结构

```python
class CostcoModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_size,  # 529条链路
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)  # 输出529个预测值
        )
```

### 在线学习核心逻辑

```python
# 1. 初始训练
X_init = matrix[:, 0:2000].T  # (2000, 529)
y_init = matrix[:, 2000]        # (529,)

for epoch in range(10):
    train(X_init, y_init)

# 2. 在线预测循环
for pred_col in range(2001, 3000):
    # 预测
    X_pred = matrix[:, pred_col-100:pred_col].T  # (100, 529)
    pred = model.predict(X_pred)  # (529,)

    # 增量更新
    X_update = matrix[:, pred_col-99:pred_col+1].T  # (100, 529)
    y_update = matrix[:, pred_col]  # (529,)

    for epoch in range(5):
        train(X_update, y_update)
```

## 💻 使用预测结果

### 加载预测结果

```python
import numpy as np

# 加载预测和真实值
predictions = np.load('output/predictions/predictions.npy')
true_values = np.load('output/predictions/true_values.npy')

print(f"预测结果形状: {predictions.shape}")  # (529, 999)
print(f"真实值形状: {true_values.shape}")      # (529, 999)
```

### 分析特定链路的预测

```python
# 查看链路0的预测
link_id = 0
link_pred = predictions[link_id, :]
link_true = true_values[link_id, :]

print(f"链路0的预测:")
print(f"  前10次: {link_pred[:10]}")
print(f"  MAE: {np.abs(link_pred - link_true).mean():.4f}")

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(link_true, label='真实值', alpha=0.7)
plt.plot(link_pred, label='预测值', alpha=0.7)
plt.title('链路0的预测 vs 真实值')
plt.xlabel('预测步骤')
plt.ylabel('流量值')
plt.legend()
plt.show()
```

### 分析所有链路的整体性能

```python
# 计算所有链路的指标
mae_per_link = np.abs(predictions - true_values).mean(axis=1)
rmse_per_link = np.sqrt(((predictions - true_values) ** 2).mean(axis=1))

print(f"所有链路的MAE:")
for link_id in range(10):
    print(f"  链路{link_id:3d}: {mae_per_link[link_id]:.4f}")

print(f"\n平均MAE: {mae_per_link.mean():.4f}")
print(f"最佳链路MAE: {mae_per_link.min():.4f} (链路{mae_per_link.argmin()})")
print(f"最差链路MAE: {mae_per_link.max():.4f} (链路{mae_per_link.argmax()})")
```

## 🔬 高级用法

### 自定义预测范围

```bash
# 只预测最后500列
# 需要修改代码中的 predict_start 和 predict_end
python online_predict_simple.py --dataset Geant_23_23_3000
```

### 不同数据集

```bash
# Abilene 数据集 (144条链路，3000列)
python convert_robust.py --dataset Abilene_12_12_3000 --time_axis col
python online_predict_simple.py --dataset Abilene_12_12_3000

# Geant 数据集 (529条链路，3000列）
python convert_robust.py --dataset Geant_23_23_3000 --time_axis col
python online_predict_simple.py --dataset Geant_23_23_3000
```

### 批量处理多个数据集

```bash
# 创建批量脚本
for dataset in Abilene_12_12_3000 Geant_23_23_3000; do
    echo "处理数据集: $dataset"
    python online_predict_simple.py --dataset $dataset
done
```

## 📈 评估指标

输出脚本会自动计算以下指标：

| 指标 | 公式 | 说明 |
|------|------|------|
| MAE | `mean(|pred - true|)` | 平均绝对误差 |
| RMSE | `sqrt(mean((pred - true)^2))` | 均方根误差 |
| NMAE | `sum(|pred - true|) / sum(|true|)` | 归一化平均绝对误差 |
| NRMSE | `sqrt(sum((pred - true)^2 / sum(true^2)))` | 归一化均方根误差 |

## 🎓 示例输出

```
======================================================================
Costco 在线学习预测
======================================================================

数据集: Geant_23_23_3000
初始窗口: 2000
滑动窗口: 100
学习率: 0.001

初始化在线预测器
======================================================================

加载矩阵: ./output/Geant_23_23_3000_matrix.npy
  形状: (529, 3000)
  链路数: 529
  时间点数: 3000
  数据范围: [0.00, 9218069948.58]

  设备: cuda

模型设置:
  隐藏层大小: 128
  学习率: 0.001
  初始窗口: 2000 (列0-1999)
  滑动窗口: 100
  训练轮数: 5 每次

======================================================================
【步骤1】初始训练
使用列 0-1999 作为历史数据
----------------------------------------------------------------------
训练数据: X形状=(2000, 529) (时间步 × 链路)
y形状=(529,) (链路)
  Epoch 1/20, Loss: 0.001234
  Epoch 5/20, Loss: 0.000987
  Epoch 10/20, Loss: 0.000876
  Epoch 15/20, Loss: 0.000812
  Epoch 20/20, Loss: 0.000798

【步骤2-4】在线预测 + 增量更新
预测列 2001 到 2999
----------------------------------------------------------------------
预测列 2100/3000, 最近100列平均损失: 1234.5678
预测列 2200/3000, 最近100列平均损失: 1235.1234
...
预测列 2999/3000, 最近100列平均损失: 1240.5678

======================================================================
在线预测完成！
预测列数: 999
======================================================================

======================================================================
预测结果评估
======================================================================

整体指标:
  MAE:  1234.5678
  RMSE: 2345.6789
  NMAE: 0.0456
  NRMSE: 0.0678

损失统计:
  平均损失: 1240.5678
  最小损失: 890.1234
  最大损失: 2345.6789
  中位数损失: 1180.4567

损失随时间变化:
  列 2001: 890.1234
  列 2002: 945.6789
  列 2003: 1012.3456
...

✓ 预测结果已保存: ./output/predictions/predictions.npy
✓ 真实值已保存: ./output/predictions/true_values.npy

======================================================================
✓ 在线学习完成！
======================================================================
```

## 📁 文件结构

```
handleData/
├── online_predict_simple.py    # ⭐ 主要在线学习脚本
├── online_learning_costco.py  # 完整版本
├── output/
│   ├── Geant_23_23_3000_matrix.npy        # (529, 3000) 输入矩阵
│   └── predictions/
│       ├── predictions.npy                # (529, 999) 预测结果
│       └── true_values.npy              # (529, 999) 真实值
```

## ⚡ 性能优化建议

1. **使用GPU**
   - 自动检测CUDA，如果可用则使用

2. **调整批量大小**
   - 可以修改代码添加batch_size参数

3. **窗口大小选择**
   - 较小窗口(50-100): 快速适应，可能过拟合
   - 中等窗口(100-200): 平衡
   - 较大窗口(200-500): 稳定，但适应性慢

4. **学习率调整**
   - 高学习率(1e-2): 快速收敛，可能不稳定
   - 中学习率(1e-3): 推荐
   - 低学习率(1e-4): 稳定，收敛慢

## 🐛 常见问题

### Q: 预测结果看起来不太准确？

A: 尝试调整：
- 增加 `--epochs` (如10或20)
- 减小 `--lr` (如1e-4)
- 增大 `--window_size` (如150或200)

### Q: 训练时间太长？

A: 尝试调整：
- 减小 `--initial_window` (如1000)
- 减小 `--window_size` (如50)
- 减小 `--epochs` (如3)

### Q: 内存不足？

A: 尝试调整：
- 减小 `--hidden_size` (如64或32)
- 减小 `--window_size`

### Q: 如何保存中间模型？

A: 可以在代码中添加：
```python
if pred_col % 100 == 0:
    torch.save(model.state_dict(), f'model_col_{pred_col}.pt')
```

## ✅ 快速执行

```bash
cd handleData

# 1. 转换数据
python convert_robust.py --dataset Geant_23_23_3000 --time_axis col

# 2. 在线学习预测
python online_predict_simple.py --dataset Geant_23_23_3000
```

完成！预测结果将保存在 `output/predictions/` 目录。
