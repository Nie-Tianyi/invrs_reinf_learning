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

### 4. 最大熵IRL (Ziebart et al., 2008) - Student3任务
- **原理**：在满足特征期望匹配约束的条件下，选择熵最大的策略分布。基于最大熵原理，假设专家行为是最随机的且与学习到的奖励函数一致。
- **数学形式**：
  - 奖励函数：r(s) = w^T φ(s)
  - Soft值迭代：V(s) = log∑_a exp(Q(s,a)/temperature)
  - 策略：π(a|s) ∝ exp(Q(s,a)/temperature)
  - 目标：最小化特征期望差异 L(w) = 0.5‖μ_expert - μ_π(w)‖² + 0.5λ‖w‖²
  - 梯度：∇L = μ_expert - μ_π(w) + λw
- **核心算法**：
  1. 从专家轨迹计算特征期望 μ_expert
  2. 初始化奖励权重 w
  3. 重复直到收敛：
     - Soft值迭代计算Q值和策略π
     - 计算策略特征期望 μ_π(w)
     - 计算梯度 ∇L = μ_expert - μ_π(w) + λw
     - 更新权重：w ← w + α∇L
- **实现文件**：`irl_algorithms.py` 中的 `maximum_entropy_irl()` 及相关辅助函数
- **关键参数**：
  - `temperature`：温度参数，控制策略随机性（温度越高策略越随机）
  - `learning_rate`：学习率，控制梯度下降步长
  - `reg_coeff`：正则化系数，防止过拟合

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

# 运行最大熵IRL（需要专家轨迹而非完整策略）
# 首先生成专家轨迹数据
expert_trajectories = env.generate_expert_dataset(
    expert_policy=expert_policy,
    n_trajectories=50,
    noise_level=0.0,
)

# 运行最大熵IRL
weights_me, reward_me, losses = maximum_entropy_irl(
    env,
    expert_trajectories,
    gamma=0.99,
    temperature=1.0,
    learning_rate=0.1,
    n_iterations=100,
    reg_coeff=0.01,
    verbose=True,
)
```

### 最大熵IRL专用演示

运行最大熵IRL专用演示脚本：
```bash
uv run python demo_maxent_irl.py
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

### `maximum_entropy_irl(env, expert_trajectories, gamma=0.99, temperature=1.0, learning_rate=0.1, n_iterations=100, reg_coeff=0.01, verbose=True)`
最大熵IRL主算法。返回奖励权重、恢复的奖励矩阵和损失历史。

### `compute_expert_feature_expectations_from_trajectories(env, trajectories, gamma=0.99)`
从专家轨迹计算特征期望。

### `soft_value_iteration(env, reward_weights, gamma=0.99, temperature=1.0, max_iter=1000, tol=1e-6)`
Soft值迭代算法，计算给定奖励权重下的soft Q值和soft value函数。

### `compute_soft_policy(Q, temperature=1.0)`
从soft Q值计算softmax策略。

### `compute_state_visitation_frequency_maxent(env, policy, gamma=0.99, initial_dist=None, max_iter=1000, tol=1e-6)`
通过反向传播计算状态访问频率（最大熵IRL专用）。

### `compute_policy_feature_expectations_maxent(env, policy, gamma=0.99, initial_dist=None)`
计算策略的特征期望（最大熵IRL专用）。

### `evaluate_reward_recovery(env, recovered_reward, true_reward)`
评估恢复的奖励与真实奖励之间的差异。

### `visualize_reward_comparison(env, true_reward, recovered_reward, title, save_path)`
可视化真实奖励与恢复奖励的对比。

## 实验观察

1. **基础线性规划IRL**：能够恢复奖励权重，但可能与真实奖励存在较大差异（高MSE，低相关性）。
2. **精确线性规划IRL**：约束较严格，在存在转移噪声的环境中可能无可行解。
3. **最大间隔IRL**：通过松弛变量提高了可行性，但恢复效果与基础版本类似。
4. **最大熵IRL**：能够较好地恢复奖励函数，特别是在有足够专家轨迹的情况下。温度参数对策略随机性有显著影响，适当调整温度可以改善恢复效果。

## 调整参数建议

- `gamma`：折扣因子，通常设为0.99
- `margin`：间隔大小，基础版本可设为1.0，精确版本可设为0.1
- `reward_bound`：奖励绝对值上限，根据真实奖励范围调整（如50.0）
- `reward_norm`：正则化类型，"l1"或"l2"

### 最大熵IRL专用参数
- `temperature`：温度参数，控制策略随机性。较低温度（0.1-0.5）产生确定性策略，较高温度（1.0-2.0）产生随机策略。
- `learning_rate`：学习率，通常设为0.01-0.1。过大可能导致不稳定，过小则收敛缓慢。
- `n_iterations`：迭代次数，通常需要100-500次迭代以获得良好收敛。
- `reg_coeff`：正则化系数，通常设为0.01-0.1，防止过拟合。

## 扩展方向

1. **深度特征版本**：使用神经网络作为特征提取器，替代手工设计特征。
2. **偏好数据集成**：结合Bradley-Terry模型，从偏好数据中学习奖励。
3. **迁移学习**：在不同环境间迁移学到的奖励函数。
4. **最大熵IRL扩展**：
   - **深度最大熵IRL**：使用深度神经网络作为奖励函数近似器。
   - **结构化最大熵IRL**：引入动作约束和状态约束，处理更复杂的决策问题。
   - **分层最大熵IRL**：在分层任务结构中应用最大熵原理。

## 文件列表

- `irl_algorithms.py` - IRL算法核心实现（包含线性规划IRL、最大间隔IRL和最大熵IRL）
- `demo_irl.py` - IRL算法综合演示脚本（包含所有算法）
- `demo_maxent_irl.py` - 最大熵IRL专用演示脚本
- `environment.py` - 网格世界环境（依赖）
- `main.py` - 主程序（生成专家数据）
- `IRL_IMPLEMENTATION.md` - 本文档

## 作者

Student2 - 线性规划、最大间隔和最大熵IRL实现
Student3 - 最大熵IRL实现

## 致谢

- Ng & Russell (2000) 的线性规划IRL算法
- Abbeel & Ng (2004) 的最大间隔IRL算法
- Ziebart et al. (2008) 的最大熵IRL算法
- CVXPY 优化库

---

*注：本实现为教学目的，实际应用中可能需要进一步调参和优化。*
