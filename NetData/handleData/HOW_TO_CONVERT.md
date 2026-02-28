# 如何将 Abilene.npy 转换为二维矩阵（每列=一个时间点）

## 方法1：使用我创建的脚本（推荐）

### 执行文件：`convert_abilene_simple.py`

这个脚本位于项目根目录，可以直接运行。

```bash
python convert_abilene_simple.py
```

**输出结果：**
- 输入：`data/Abilene.npy` (12, 12, 3000)
- 输出：`handleData/output/Abilene_matrix_time_user_item.npy` (3000, 144)

**矩阵说明：**
- 每一列（共144列）代表一个用户-物品对
- 每一行（共3000行）代表一个时间点
- 所以"每一列表示一个时间点"的理解是：矩阵的**行**是时间点

### 方法2：使用 tensor_converter.py

在 `handleData` 文件夹中：

```python
from tensor_converter import TensorConverter

# 创建转换器
converter = TensorConverter('Abilene')

# 转换为矩阵（每列=一个时间点）
matrix = converter.convert_to_matrix(mode='time_user_item', save=True)
```

### 方法3：手动转换（最简单）

如果你想自己写代码，可以使用以下简洁版本：

```python
import numpy as np

# 1. 加载原始张量
tensor = np.load('./data/Abilene.npy')

# 2. 转换形状: (12, 12, 3000) -> (3000, 144)
# 每列代表一个时间点
num_user, num_item, num_time = tensor.shape
matrix = tensor.transpose(2, 0, 1).reshape(num_time, num_user * num_item)

# 3. 保存矩阵
np.save('./handleData/output/Abilene_matrix_time_user_item.npy', matrix)

print(f"原始形状: {tensor.shape}")
print(f"转换后形状: {matrix.shape}")
print(f"保存成功！")
```

## 各种转换模式对比

| 模式 | 代码 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| **每列=时间点** | `time_user_item` | (12,12,3000) | (3000, 144) | 每行是时间点快照 |
| 展平 | `flatten` | (12,12,3000) | (144, 3000) | 每行是用户-物品对的时间序列 |
| 按用户 | `user_time` | (12,12,3000) | (12, 36000) | 每行是一个用户 |
| 按物品 | `item_time` | (12,12,3000) | (12, 36000) | 每行是一个物品 |
| 平均 | `avg` | (12,12,3000) | (12, 12) | 用户-物品关系矩阵 |

## 验证转换结果

```python
import numpy as np

# 加载转换后的矩阵
matrix = np.load('./handleData/output/Abilene_matrix_time_user_item.npy')

print(f"矩阵形状: {matrix.shape}")
# 输出: (3000, 144)

print(f"第一行（时间点0）的数据: {matrix[0]}")
print(f"最后一行（时间点2999）的数据: {matrix[-1]}")

# 验证数据完整性
print(f"数据范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
print(f"非零值数量: {(matrix != 0).sum()}")
```

## 在训练中使用转换后的矩阵

```python
import numpy as np
import torch

# 1. 加载矩阵
matrix = np.load('./handleData/output/Abilene_matrix_time_user_item.npy')

# 2. 转换为 PyTorch 张量
data = torch.from_numpy(matrix).float()

# 3. 如果有CUDA，移到GPU
if torch.cuda.is_available():
    data = data.cuda()

# 4. 归一化
max_val = data.max()
data_normalized = data / max_val

# 5. 现在可以使用这个矩阵进行训练
print(f"训练数据形状: {data_normalized.shape}")
# (3000, 144) - 3000个时间点，每个时间点有144个特征
```

## 常见问题

### Q: 为什么矩阵形状是 (3000, 144) 而不是 (144, 3000)？

A: 因为你要"每一列表示一个时间点"，所以：
- 行数 = 时间点数 = 3000
- 列数 = 用户数 × 物品数 = 12 × 12 = 144

### Q: 如何理解这个矩阵？

A:
- 矩阵的每一行代表一个时间点的快照
- 矩阵的每一列代表一个特定的用户-物品对
- 比如：第5行第10列的值 = 在时间点5，用户0物品1的值

### Q: 如何加载已转换的矩阵？

A:
```python
import numpy as np
matrix = np.load('./handleData/output/Abilene_matrix_time_user_item.npy')
```

## 推荐执行步骤

1. **打开命令行/终端**
2. **进入项目目录**
   ```bash
   cd NetData
   ```
3. **运行转换脚本**
   ```bash
   python convert_abilene_simple.py
   ```
4. **检查输出**
   - 输出文件：`handleData/output/Abilene_matrix_time_user_item.npy`
   - 形状：(3000, 144)

## 完整示例代码

如果 `convert_abilene_simple.py` 无法运行，你可以手动执行以下代码：

```python
# 创建一个新文件 convert_now.py
import numpy as np
import os

# 设置路径
tensor_file = './data/Abilene.npy'
output_dir = './handleData/output'
output_file = 'Abilene_matrix_time_user_item.npy'

# 加载张量
print("加载张量...")
tensor = np.load(tensor_file)
print(f"原始张量形状: {tensor.shape}")

# 转换为矩阵（每列=一个时间点）
print("转换中...")
num_user, num_item, num_time = tensor.shape
matrix = tensor.transpose(2, 0, 1).reshape(num_time, num_user * num_item)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 保存矩阵
output_path = os.path.join(output_dir, output_file)
np.save(output_path, matrix)

print(f"\n转换完成！")
print(f"输出矩阵形状: {matrix.shape}")
print(f"输出文件: {output_path}")
print(f"数据范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
```

保存为 `convert_now.py`，然后运行：
```bash
python convert_now.py
```

---

**总结：要完成转换，请执行以下任一文件：**

1. ✅ `convert_abilene_simple.py` - 最简单，在项目根目录
2. ✅ `handleData/convert_abilene.py` - 在 handleData 文件夹
3. ✅ 手动运行上面的完整示例代码

转换完成后，矩阵将保存在 `handleData/output/Abilene_matrix_time_user_item.npy`
