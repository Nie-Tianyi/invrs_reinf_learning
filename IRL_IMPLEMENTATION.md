# 逆强化学习算法实现（Student2）

## 概述

本模块实现了两种逆强化学习（IRL）算法：
1. **线性规划IRL** (Ng & Russell, 2000) - 基础版本
2. **最大间隔IRL** (Abbeel & Ng, 2004)

## 算法描述

### 1. 线性规划IRL (基础版本)
- **原理**：通过线性规划求解奖励权重，使得专家策略的特征期望优于随机采样策略的特征期望
- **数学形式**：
  - 目标：最小化权重范数（L1或L2正则化）
  - 约束：w^T μ_expert ≥ w^T μ_random + margin
- **实现文件**：`irl_algorithms.py` 中的 `linear_programming_irl()` 函数

### 2. 线性规划IRL (Ng & Russell 精确版本)
- **原理**：基于Ng & Russell (2000) 的精确线性规划公式
- **数学形式**：
  - 目标：最大化奖励和 ∑_s r(s)
  - 约束：(P_{π_E} - P_a)(I - γ P_{π_E})^{-1} r ≥ margin, ∀s,a ≠ π_E(s)
  - 边界：|r(s)| ≤ reward_bound
- **实现文件**：`irl_algorithms.py` 中的 `linear_programming_irl_ng_russell()` 函数

### 3. 最大间隔IRL
- **原理**：最大化专家策略与所有其他策略之间的特征期望间隔，使用松弛变量处理不可行情况
- **数学形式**：
  - 目标：最小化权重范数 + C * ∑ ξ_i
  - 约束：w^T μ_expert ≥ w^T μ_random + margin - ξ_i
- **实现文件**：`irl_algorithms.py` 中的 `maximum_margin_irl()` 函数

## 依赖项

- numpy >= 2.4.3
- matplotlib >= 3.10.8
- cvxpy >= 1.5.0

已通过 `uv sync` 安装。

## 使用方法

### 基本用法

```python
from environment import AdvancedGridWorld, value_iteration
from irl_algorithms import linear_programming_irl, maximum_margin_irl

# 初始化环境
env = AdvancedGridWorld()

# 生成专家策略
expert_policy, _ = value_iteration(env)

# 运行线性规划IRL
weights_lp, reward_lp = linear_programming_irl(env, expert_policy)

# 运行最大间隔IRL
weights_mm, reward_mm = maximum_margin_irl(env, expert_policy)
```

### 评估恢复效果

```python
from irl_algorithms import evaluate_reward_recovery, visualize_reward_comparison

true_reward = env.get_true_reward()
metrics = evaluate_reward_recovery(env, reward_lp, true_reward)
print(f"MSE: {metrics['mse']}, 相关性: {metrics['correlation']}")

# 可视化对比
visualize_reward_comparison(env, true_reward, reward_lp, "恢复效果对比")
```

### 完整演示

运行演示脚本：
```bash
uv run python demo_irl.py
```

## 关键接口

### `compute_feature_expectations(env, policy, gamma=0.99)`
计算给定策略下的折扣特征期望向量。

### `linear_programming_irl(env, expert_policy, gamma=0.99, margin=1.0, reward_norm="l2")`
线性规划IRL基础版本。

### `linear_programming_irl_ng_russell(env, expert_policy, gamma=0.99, margin=1.0, reward_bound=1.0)`
线性规划IRL精确版本（Ng & Russell）。

### `maximum_margin_irl(env, expert_policy, gamma=0.99, margin=1.0, reward_norm="l2")`
最大间隔IRL。

### `evaluate_reward_recovery(env, recovered_reward, true_reward)`
评估恢复的奖励与真实奖励之间的差异。

### `visualize_reward_comparison(env, true_reward, recovered_reward, title, save_path)`
可视化真实奖励与恢复奖励的对比。

## 实验观察

1. **基础线性规划IRL**：能够恢复奖励权重，但可能与真实奖励存在较大差异（高MSE，低相关性）。
2. **精确线性规划IRL**：约束较严格，在存在转移噪声的环境中可能无可行解。
3. **最大间隔IRL**：通过松弛变量提高了可行性，但恢复效果与基础版本类似。

## 调整参数建议

- `gamma`：折扣因子，通常设为0.99
- `margin`：间隔大小，基础版本可设为1.0，精确版本可设为0.1
- `reward_bound`：奖励绝对值上限，根据真实奖励范围调整（如50.0）
- `reward_norm`：正则化类型，"l1"或"l2"

## 扩展方向

1. **深度特征版本**：使用神经网络作为特征提取器，替代手工设计特征。
2. **偏好数据集成**：结合Bradley-Terry模型，从偏好数据中学习奖励。
3. **迁移学习**：在不同环境间迁移学到的奖励函数。

## 文件列表

- `irl_algorithms.py` - IRL算法核心实现
- `demo_irl.py` - 演示脚本
- `environment.py` - 网格世界环境（依赖）
- `main.py` - 主程序（生成专家数据）
- `IRL_IMPLEMENTATION.md` - 本文档

## 作者

Student2 - 线性规划和最大间隔IRL实现

## 致谢

- Ng & Russell (2000) 的线性规划IRL算法
- Abbeel & Ng (2004) 的最大间隔IRL算法
- CVXPY 优化库

---

*注：本实现为教学目的，实际应用中可能需要进一步调参和优化。*
