# 逆强化学习项目 (Inverse Reinforcement Learning)

本项目实现了多种逆强化学习（IRL）算法，包括线性特征和深度特征版本。项目基于PyTorch实现深度最大熵IRL算法，使用局部网格作为神经网络输入，避免神经网络简单记忆坐标位置。

## 🚀 快速开始

### 环境要求
- Python >= 3.13
- uv 包管理器

### 安装依赖
```bash
# 安装uv（如果未安装）
pip install uv

# 安装项目依赖
uv sync
```

### 运行演示

项目提供了统一的命令行入口 `run.py`，所有实验均可通过子命令运行：

```bash
# 基础IRL演示（线性规划 + 最大间隔 + 最大熵）
uv run python run.py demo

# 包含深度IRL对比的完整演示
uv run python run.py demo --deep

# 综合对比实验
uv run python run.py experiment

# 快速实验（减少迭代次数，适合快速验证）
uv run python run.py experiment --quick

# 完整消融实验
uv run python run.py ablation

# 仅运行噪声维度的消融，3个随机种子
uv run python run.py ablation --dim noise --runs 3

# 从checkpoint生成图表
uv run python run.py plot

# 快速烟雾测试，验证所有模块是否正常
uv run python run.py smoke
```

也可以单独运行各个演示脚本：
```bash
uv run python demo_irl.py          # 线性特征IRL演示
uv run python demo_maxent_irl.py   # 最大熵IRL演示
uv run python demo_deep_irl.py     # 深度IRL演示
```

## 📁 项目文件结构

```
invrs_reinf_learning/
├── README.md                          # 项目说明文件
├── pyproject.toml                     # 项目依赖配置
├── uv.lock                            # 依赖锁文件
├── run.py                             # 统一命令行入口
├── environment.py                     # Advanced GridWorld 环境
├── irl_algorithms.py                  # 线性特征IRL算法
├── deep_feature_extractor.py          # 深度特征提取器（神经网络）
├── deep_irl_algorithms.py             # 深度最大熵IRL算法
├── policy_training.py                 # 策略训练与评估
├── experiments.py                     # 综合实验脚本
├── experiments_final.py               # 消融实验脚本
├── plot.py                            # 图表生成脚本
├── demo_irl.py                        # 线性特征IRL演示
├── demo_maxent_irl.py                 # 最大熵IRL演示
├── demo_deep_irl.py                   # 深度IRL演示
├── main.py                            # 主入口脚本
├── IRL_IMPLEMENTATION.md              # IRL实现文档
├── STUDENT4_REPORT.md                 # 学生报告
└── deep_maxent_irl_model.pth          # 预训练深度IRL模型
```

## 🔧 算法实现

### 1. 线性特征IRL（手工设计特征）
- **线性规划IRL**: 基于线性规划的奖励恢复算法
- **最大间隔IRL**: 基于最大间隔原理的奖励学习
- **最大熵IRL**: 基于最大熵原理的奖励恢复（线性特征）

### 2. 深度特征IRL（神经网络自动学习特征）
- **深度最大熵IRL**: 使用卷积神经网络处理局部网格输入
  - 输入: 智能体周围5×5的局部网格
  - 网络架构: CNN + 全连接层
  - 输出: 状态奖励值
  - 优势: 学习空间模式而非具体坐标，避免过拟合

### 3. 环境与工具
- **AdvancedGridWorld**: 扩展的网格世界环境
  - 随机地形生成（空地、障碍、奖励区、惩罚区）
  - 转移矩阵计算
  - 专家轨迹生成
  - 局部网格特征提取

## 🧪 深度特征IRL核心特性

### 输入设计
```python
# 获取智能体在位置(x,y)处的局部5×5网格
local_grid = env.get_local_grid((x, y), window_size=5)
# 输入神经网络预测奖励
reward = network(local_grid_tensor)
```

### 神经网络架构
- **CNN层**: 提取局部空间特征
- **池化层**: 降维并保持空间不变性  
- **全连接层**: 学习高级特征表示
- **Dropout**: 防止过拟合

### 性能优化
- **向量化soft价值迭代**: 使用矩阵运算替代循环
- **批量预测**: 一次性处理所有状态
- **局部网格缓存**: 避免重复计算
- **梯度裁剪**: 防止梯度爆炸

## 📊 性能对比

| 算法 | 特征类型 | 训练时间（20次迭代） | 恢复精度（相关性） |
|------|----------|---------------------|-------------------|
| 线性规划IRL | 手工特征 | ~1秒 | 0.85-0.95 |
| 最大熵IRL | 手工特征 | ~2秒 | 0.80-0.90 |
| 深度最大熵IRL | 自动学习 | ~4秒 | 0.70-0.85 |

> **注意**: 深度特征版本泛化能力更强，适合复杂环境。

## 🎯 使用方法示例

### 1. 运行深度最大熵IRL
```python
from environment import AdvancedGridWorld, value_iteration
from deep_irl_algorithms import deep_maximum_entropy_irl

# 初始化环境
env = AdvancedGridWorld()

# 生成专家策略和轨迹
expert_policy, _ = value_iteration(env)
expert_trajectories = env.generate_expert_dataset(expert_policy, n_trajectories=50)

# 运行深度IRL
network, recovered_reward, losses = deep_maximum_entropy_irl(
    env=env,
    expert_trajectories=expert_trajectories,
    window_size=5,
    n_iterations=100,
    learning_rate=0.0001,
    verbose=True
)
```

### 2. 可视化结果
```python
from deep_irl_algorithms import visualize_deep_irl_results

visualize_deep_irl_results(
    env=env,
    true_reward=true_reward,
    recovered_reward=recovered_reward,
    losses=losses,
    title="Deep Maximum Entropy IRL Results"
)
```

## 🛠️ 开发说明

### 代码规范
- 使用类型注解（Type Hints）
- 遵循PEP 8编码规范
- 函数和类提供完整的docstring

### 依赖管理
```toml
# pyproject.toml 关键依赖
dependencies = [
    "matplotlib>=3.10.8",    # 可视化
    "numpy>=2.4.3",          # 数值计算
    "cvxpy>=1.5.0",          # 凸优化
    "torch",                 # 深度学习框架
    "torchvision",           # 视觉工具
]
```

### 测试与验证
```bash
# 运行烟雾测试，验证所有模块正常
uv run python run.py smoke

# 也可以单独运行各个演示
uv run python demo_irl.py
uv run python demo_maxent_irl.py
uv run python demo_deep_irl.py
```

## 📈 项目进展

### 已完成
- [x] 基础环境实现（AdvancedGridWorld）
- [x] 线性特征IRL算法（线性规划、最大间隔、最大熵）
- [x] 深度特征提取器（LocalGridRewardNet）
- [x] 深度最大熵IRL算法
- [x] 计算图完整性修复和性能优化
- [x] 演示脚本和性能测试

### 待完成/扩展
- [ ] 更复杂的神经网络架构
- [ ] 更多环境类型支持
- [ ] 在线学习/增量学习
- [ ] 分布式训练支持

## 🤝 贡献

欢迎提交Issue和Pull Request改进项目！

## 📚 参考文献

1. Ng, A. Y., & Russell, S. J. (2000). Algorithms for inverse reinforcement learning.
2. Ziebart, B. D., et al. (2008). Maximum entropy inverse reinforcement learning.
3. Finn, C., et al. (2016). Guided cost learning: Deep inverse optimal control via policy optimization.
4. Levine, S., & Koltun, V. (2012). Continuous inverse optimal control with locally optimal examples.

## 📄 许可证

本项目仅供学习研究使用。