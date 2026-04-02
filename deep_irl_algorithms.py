"""
深度逆强化学习算法实现
基于PyTorch的深度最大熵IRL（Deep Maximum Entropy IRL）

使用局部网格作为输入，神经网络预测奖励，通过最大熵原理学习奖励函数。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from deep_feature_extractor import LocalGridRewardNet, DeepFeatureExtractor
from environment import AdvancedGridWorld,value_iteration


def compute_reward_matrix_from_network(
    network: nn.Module,
    env: AdvancedGridWorld,
    window_size: int = 5,
    device: Optional[str] = None
) -> np.ndarray:
    """
    使用神经网络计算整个环境的奖励矩阵
    
    :param network: 奖励预测神经网络
    :param env: 环境对象
    :param window_size: 局部网格窗口大小
    :param device: 计算设备（如为None，则使用网络所在的设备）
    :return: 奖励矩阵，形状 (grid_size, grid_size)
    """
    network.eval()
    reward_matrix = np.zeros((env.grid_size, env.grid_size))
    
    # 获取网络所在的设备
    network_device = next(network.parameters()).device
    if device is not None:
        # 如果指定了设备，将网络移动到该设备（临时）
        network.to(device)
        network_device = device
    
    with torch.no_grad():
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                # 获取局部网格
                local_grid = env.get_local_grid((x, y), window_size=window_size)
                local_grid_tensor = torch.tensor(
                    local_grid, dtype=torch.float32
                ).unsqueeze(0).to(network_device)
                
                # 预测奖励
                reward = network(local_grid_tensor).cpu().item()
                reward_matrix[x, y] = reward
    
    # 如果之前移动了网络，将其移回原设备（但通常不需要）
    if device is not None and device != str(next(network.parameters()).device):
        # 实际上我们不应该移动网络，但这里只是保险
        pass
    
    return reward_matrix


def compute_reward_tensor_from_network(
    network: nn.Module,
    env: AdvancedGridWorld,
    window_size: int = 5,
    device: Optional[str] = None,
    local_grids_cache: Optional[List[np.ndarray]] = None,
) -> torch.Tensor:
    """
    使用神经网络计算整个环境的奖励张量（保留计算图，批量处理版本）
    
    :param network: 奖励预测神经网络
    :param env: 环境对象
    :param window_size: 局部网格窗口大小
    :param device: 计算设备（如为None，则使用网络所在的设备）
    :param local_grids_cache: 预计算的局部网格缓存（如为None则从环境获取）
    :return: 奖励张量，形状 (n_states,)，需要梯度
    """
    network.train()  # 保持训练模式以保留梯度
    
    # 获取网络所在的设备
    network_device = next(network.parameters()).device
    if device is not None:
        # 如果指定了设备，将网络移动到该设备
        network.to(device)
        network_device = device
    
    # 使用缓存或收集局部网格
    if local_grids_cache is not None:
        local_grids = local_grids_cache
    else:
        # 批量收集所有局部网格
        local_grids = []
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                local_grid = env.get_local_grid((x, y), window_size=window_size)
                local_grids.append(local_grid)
    
    # 批量转换为张量（一次性操作，提高效率）
    # 形状: (n_states, window_size, window_size)
    local_grids_tensor = torch.tensor(
        np.array(local_grids), dtype=torch.float32, device=network_device
    )
    
    # 批量预测奖励（保留计算图）
    # 网络期望输入形状: (batch_size, window_size, window_size)
    rewards = network(local_grids_tensor).squeeze(-1)  # 形状: (n_states,)
    
    return rewards


def soft_value_iteration_with_reward_matrix(
    env: AdvancedGridWorld,
    reward_matrix: np.ndarray,
    gamma: float = 0.99,
    temperature: float = 1.0,
    theta: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用奖励矩阵进行soft价值迭代
    
    :param env: 环境对象
    :param reward_matrix: 奖励矩阵 (grid_size, grid_size)
    :param gamma: 折扣因子
    :param temperature: 温度参数（控制策略随机性）
    :param theta: 收敛阈值
    :param max_iter: 最大迭代次数
    :return: Soft Q值矩阵 (n_states, n_actions)，Soft 价值函数 (n_states,)
    """
    n_states = env.n_states
    n_actions = env.n_actions
    P = env.transition_matrix
    R = reward_matrix.flatten()  # 一维奖励向量
    
    # 初始化价值函数
    V = np.zeros(n_states)
    
    # Soft 价值迭代主循环
    for iteration in range(max_iter):
        delta = 0
        
        # 对每个状态更新价值
        for s in range(n_states):
            old_v = V[s]
            
            # 计算每个动作的Q值
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = np.sum(P[s, a] * (R + gamma * V))
            
            # Softmax over actions (soft value)
            # V(s) = temperature * log(∑_a exp(Q(s,a)/temperature))
            if temperature > 0:
                exp_q = np.exp(q_values / temperature)
                V[s] = temperature * np.log(np.sum(exp_q) + 1e-10)
            else:
                V[s] = np.max(q_values)  # 退化为标准价值迭代
            
            delta = max(delta, abs(old_v - V[s]))
        
        # 收敛判断
        if delta < theta:
            break
    
    # 计算Soft Q值
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = np.sum(P[s, a] * (R + gamma * V))
    
    return Q, V


def soft_value_iteration_with_reward_tensor(
    env: AdvancedGridWorld,
    reward_tensor: torch.Tensor,
    gamma: float = 0.99,
    temperature: float = 1.0,
    theta: float = 1e-6,
    max_iter: int = 1000,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用奖励张量进行soft价值迭代（PyTorch版本，保留计算图）
    
    :param env: 环境对象
    :param reward_tensor: 奖励张量，形状 (n_states,)
    :param gamma: 折扣因子
    :param temperature: 温度参数（控制策略随机性）
    :param theta: 收敛阈值
    :param max_iter: 最大迭代次数
    :param device: 计算设备（如为None，则使用reward_tensor的设备）
    :return: Soft Q值张量 (n_states, n_actions)，Soft 价值函数张量 (n_states,)
    """
    # 确定设备：优先使用reward_tensor的设备
    if device is None:
        device = reward_tensor.device
    else:
        # 如果指定了设备，将reward_tensor移动到该设备（如果尚未在）
        if reward_tensor.device != torch.device(device):
            reward_tensor = reward_tensor.to(device)
    
    n_states = env.n_states
    n_actions = env.n_actions
    
    # 将转移矩阵转换为PyTorch张量（移动到相同设备）
    P = torch.tensor(env.transition_matrix, dtype=torch.float32, device=device)
    R = reward_tensor  # 形状 (n_states,)
    
    # 初始化价值函数
    V = torch.zeros(n_states, device=device)
    
    # Soft 价值迭代主循环（向量化版本）
    for iteration in range(max_iter):
        # 计算期望奖励：E[R|s,a] = Σ_s' P(s'|s,a) * R(s')
        # P形状: (n_states, n_actions, n_states), R形状: (n_states,)
        # 结果形状: (n_states, n_actions)
        expected_reward = torch.matmul(P, R.unsqueeze(-1)).squeeze(-1)
        
        # 计算期望下一个状态的价值：E[V(s')|s,a] = Σ_s' P(s'|s,a) * V(s')
        expected_next_value = torch.matmul(P, V.unsqueeze(-1)).squeeze(-1)
        
        # 计算Q值：Q(s,a) = E[R|s,a] + gamma * E[V(s')|s,a]
        Q = expected_reward + gamma * expected_next_value
        
        # 保存旧的V值用于收敛判断
        V_old = V.clone()
        
        # 更新价值函数（向量化）
        if temperature > 0:
            # Softmax value: V(s) = temperature * log(∑_a exp(Q(s,a)/temperature))
            # 使用logsumexp提高数值稳定性
            V = temperature * torch.logsumexp(Q / temperature, dim=1)
        else:
            # 确定性价值迭代：V(s) = max_a Q(s,a)
            V, _ = torch.max(Q, dim=1)
        
        # 计算最大变化量
        delta = torch.max(torch.abs(V - V_old)).item()
        
        # 收敛判断
        if delta < theta:
            break
    
    # 重新计算最终的Q值（确保使用最终的V）
    expected_reward = torch.matmul(P, R.unsqueeze(-1)).squeeze(-1)
    expected_next_value = torch.matmul(P, V.unsqueeze(-1)).squeeze(-1)
    Q = expected_reward + gamma * expected_next_value
    
    return Q, V


def compute_soft_policy_from_q(
    Q: np.ndarray,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    从Soft Q值计算softmax策略（支持NumPy数组和PyTorch张量）
    
    :param Q: Soft Q值矩阵 (n_states, n_actions)，可以是np.ndarray或torch.Tensor
    :param temperature: 温度参数
    :param device: 计算设备（如果Q是np.ndarray）
    :return: 策略矩阵 (n_states, n_actions) 作为torch.Tensor
    """
    # 将Q转换为torch.Tensor（如果需要）
    if isinstance(Q, np.ndarray):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Q_tensor = torch.tensor(Q, dtype=torch.float32, device=device)
    else:
        Q_tensor = Q
        device = Q.device
    
    if temperature <= 0:
        # 确定性策略
        policy = torch.zeros_like(Q_tensor)
        best_actions = torch.argmax(Q_tensor, dim=1)
        policy[torch.arange(Q_tensor.shape[0]), best_actions] = 1.0
        return policy
    
    # Softmax策略（使用稳定的softmax实现）
    policy = torch.softmax(Q_tensor / temperature, dim=1)
    return policy


def compute_trajectory_log_likelihood(
    trajectory: Dict,
    Q: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.99,
    env: Optional[AdvancedGridWorld] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    计算轨迹的对数似然（最大熵IRL核心）
    
    :param trajectory: 轨迹字典，包含'states'和'actions'
    :param Q: Soft Q值矩阵
    :param policy: 策略矩阵（可以是np.ndarray或torch.Tensor）
    :param gamma: 折扣因子
    :param env: 环境对象（用于状态索引转换）
    :param device: 计算设备（如果policy是torch.Tensor）
    :return: 轨迹的对数似然（torch.Tensor标量）
    """
    states = trajectory['states']
    actions = trajectory['actions']
    
    # 将policy转换为torch.Tensor（如果需要）
    if isinstance(policy, np.ndarray):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy_tensor = torch.tensor(policy, dtype=torch.float32, device=device)
    else:
        policy_tensor = policy
    
    log_likelihood = torch.tensor(0.0, device=policy_tensor.device)
    
    for t, (state, action) in enumerate(zip(states, actions)):
        if env is not None:
            s = env._state_to_idx(state)
        else:
            # 假设状态已经是索引
            s = state
        
        # 动作概率
        action_prob = policy_tensor[s, action]
        
        # 折扣因子
        discount = gamma ** t
        
        log_likelihood = log_likelihood + discount * torch.log(action_prob + 1e-10)
    
    return log_likelihood


def deep_maximum_entropy_irl(
    env: AdvancedGridWorld,
    expert_trajectories: List[Dict],
    network: Optional[nn.Module] = None,
    window_size: int = 5,
    gamma: float = 0.99,
    temperature: float = 1.0,
    learning_rate: float = 0.001,
    n_iterations: int = 200,
    reg_coeff: float = 0.01,
    batch_size: Optional[int] = None,
    verbose: bool = True,
    device: Optional[str] = None,
) -> Tuple[nn.Module, np.ndarray, List[float]]:
    """
    深度最大熵IRL主算法
    
    :param env: AdvancedGridWorld环境
    :param expert_trajectories: 专家轨迹列表
    :param network: 预训练的神经网络（如为None则创建默认网络）
    :param window_size: 局部网格窗口大小
    :param gamma: 折扣因子
    :param temperature: 温度参数
    :param learning_rate: 学习率
    :param n_iterations: 迭代次数
    :param reg_coeff: 正则化系数（L2正则化）
    :param batch_size: 批量大小（如为None则使用全批量）
    :param verbose: 是否打印训练信息
    :param device: 计算设备
    :return: 训练好的神经网络, 恢复的奖励矩阵, 损失历史
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 初始化神经网络
    if network is None:
        network = LocalGridRewardNet(
            window_size=window_size,
            n_terrain_types=5,
            hidden_dims=(128, 64, 32),
            dropout_rate=0.2,
            use_cnn=True,
        )
    network = network.to(device)
    
    # 2. 准备优化器
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=reg_coeff)
    
    # 3. 准备训练数据
    # 提取所有状态用于批量处理
    all_states = []
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            all_states.append((x, y))
    
    # 3.1 预先计算局部网格缓存（环境不变，可重复使用）
    local_grids_cache = []
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            local_grid = env.get_local_grid((x, y), window_size=window_size)
            local_grids_cache.append(local_grid)
    
    # 4. 训练循环
    losses = []
    
    for iteration in range(n_iterations):
        network.train()
        optimizer.zero_grad()
        
        # 4.1 计算当前神经网络下的奖励张量（保留计算图，使用缓存）
        reward_tensor = compute_reward_tensor_from_network(
            network, env, window_size=window_size, device=device,
            local_grids_cache=local_grids_cache
        )
        
        # 4.2 计算soft Q值和策略（使用PyTorch张量）
        Q, V = soft_value_iteration_with_reward_tensor(
            env, reward_tensor, gamma=gamma, temperature=temperature,
            device=device
        )
        policy = compute_soft_policy_from_q(Q, temperature=temperature, device=device)
        
        # 4.3 计算负对数似然损失（使用张量）
        neg_log_likelihood = torch.tensor(0.0, device=device)
        for traj in expert_trajectories:
            log_likelihood = compute_trajectory_log_likelihood(
                traj, Q, policy, gamma=gamma, env=env, device=device
            )
            neg_log_likelihood = neg_log_likelihood - log_likelihood
        
        # 平均损失
        loss = neg_log_likelihood / len(expert_trajectories)
        
        # 4.4 反向传播
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            # 计算当前奖励张量的范围
            reward_np = reward_tensor.detach().cpu().numpy()
            reward_min = reward_np.min()
            reward_max = reward_np.max()
            reward_mean = reward_np.mean()
            
            print(f"Iteration {iteration:3d}: "
                  f"loss={loss.item():.6f}, "
                  f"reward=[{reward_min:.2f}, {reward_max:.2f}], "
                  f"mean={reward_mean:.2f}")
    
    # 5. 计算最终的奖励矩阵
    network.eval()
    final_reward_matrix = compute_reward_matrix_from_network(
        network, env, window_size=window_size, device='cpu'
    )
    
    return network, final_reward_matrix, losses


def evaluate_deep_irl_recovery(
    env: AdvancedGridWorld,
    recovered_reward: np.ndarray,
    true_reward: np.ndarray,
) -> Dict[str, float]:
    """
    评估深度IRL恢复的奖励与真实奖励之间的差异
    
    :param env: 环境对象
    :param recovered_reward: 恢复的奖励矩阵
    :param true_reward: 真实奖励矩阵
    :return: 评估指标字典
    """
    # 确保形状一致
    assert recovered_reward.shape == true_reward.shape
    
    # 计算各种指标
    mse = np.mean((recovered_reward - true_reward) ** 2)
    mae = np.mean(np.abs(recovered_reward - true_reward))
    correlation = np.corrcoef(recovered_reward.flatten(), true_reward.flatten())[0, 1]
    
    # 归一化后的余弦相似度
    recovered_norm = recovered_reward / (np.linalg.norm(recovered_reward) + 1e-10)
    true_norm = true_reward / (np.linalg.norm(true_reward) + 1e-10)
    cosine_sim = np.dot(recovered_norm.flatten(), true_norm.flatten())
    
    # 奖励分布统计
    recovered_stats = {
        'min': recovered_reward.min(),
        'max': recovered_reward.max(),
        'mean': recovered_reward.mean(),
        'std': recovered_reward.std(),
    }
    
    true_stats = {
        'min': true_reward.min(),
        'max': true_reward.max(),
        'mean': true_reward.mean(),
        'std': true_reward.std(),
    }
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        "cosine_similarity": cosine_sim,
        "recovered_stats": recovered_stats,
        "true_stats": true_stats,
    }


def visualize_deep_irl_results(
    env: AdvancedGridWorld,
    true_reward: np.ndarray,
    recovered_reward: np.ndarray,
    losses: List[float],
    title: str = "Deep Maximum Entropy IRL Results",
    save_path: Optional[str] = None,
):
    """
    可视化深度IRL结果
    
    :param env: 环境对象
    :param true_reward: 真实奖励矩阵
    :param recovered_reward: 恢复的奖励矩阵
    :param losses: 训练损失历史
    :param title: 图表标题
    :param save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 真实奖励热力图
    im1 = axes[0, 0].imshow(true_reward, cmap="RdYlGn", origin="upper")
    axes[0, 0].set_title("Ground Truth Reward", fontsize=14)
    axes[0, 0].set_xlabel("Y Coordinate")
    axes[0, 0].set_ylabel("X Coordinate")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 恢复奖励热力图
    im2 = axes[0, 1].imshow(recovered_reward, cmap="RdYlGn", origin="upper")
    axes[0, 1].set_title("Recovered Reward (Deep IRL)", fontsize=14)
    axes[0, 1].set_xlabel("Y Coordinate")
    axes[0, 1].set_ylabel("X Coordinate")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 差异热力图
    diff = recovered_reward - true_reward
    im3 = axes[0, 2].imshow(diff, cmap="RdBu", origin="upper", vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes[0, 2].set_title("Difference (Recovered - True)", fontsize=14)
    axes[0, 2].set_xlabel("Y Coordinate")
    axes[0, 2].set_ylabel("X Coordinate")
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. 训练损失曲线
    axes[1, 0].plot(losses)
    axes[1, 0].set_title("Training Loss", fontsize=14)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Negative Log-Likelihood")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 奖励值分布直方图
    axes[1, 1].hist(true_reward.flatten(), bins=30, alpha=0.5, label="True", color="blue")
    axes[1, 1].hist(recovered_reward.flatten(), bins=30, alpha=0.5, label="Recovered", color="red")
    axes[1, 1].set_title("Reward Distribution", fontsize=14)
    axes[1, 1].set_xlabel("Reward Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 散点图：真实vs恢复奖励
    axes[1, 2].scatter(true_reward.flatten(), recovered_reward.flatten(), alpha=0.5, s=10)
    axes[1, 2].plot([true_reward.min(), true_reward.max()], 
                    [true_reward.min(), true_reward.max()], 
                    'r--', alpha=0.5, label="y=x")
    axes[1, 2].set_title("True vs Recovered Reward", fontsize=14)
    axes[1, 2].set_xlabel("True Reward")
    axes[1, 2].set_ylabel("Recovered Reward")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def test_deep_maximum_entropy_irl():
    """测试深度最大熵IRL算法"""
    print("=" * 70)
    print("深度最大熵IRL算法测试")
    print("=" * 70)
    
    # 1. 初始化环境
    print("\n1. 初始化Advanced GridWorld环境")
    env = AdvancedGridWorld()
    print(f"   网格大小: {env.grid_size}x{env.grid_size}")
    print(f"   总状态数: {env.n_states}")
    
    # 2. 生成专家最优策略
    print("\n2. 生成专家最优策略")
    expert_policy, expert_V = value_iteration(env)
    print(f"   专家策略形状: {expert_policy.shape}")
    
    # 3. 生成专家轨迹数据
    print("\n3. 生成专家轨迹数据")
    expert_trajectories = env.generate_expert_dataset(
        expert_policy=expert_policy,
        n_trajectories=50,  # 使用50条轨迹
        noise_level=0.0,
    )
    print(f"   轨迹数量: {len(expert_trajectories)}")
    print(f"   平均轨迹长度: {np.mean([len(traj['states']) for traj in expert_trajectories]):.1f}")
    
    # 4. 获取真实奖励矩阵
    true_reward = env.get_true_reward()
    
    # 5. 运行深度最大熵IRL
    print("\n4. 运行深度最大熵IRL算法")
    print("   开始训练...")
    
    network, recovered_reward, losses = deep_maximum_entropy_irl(
        env=env,
        expert_trajectories=expert_trajectories,
        network=None,
        window_size=5,
        gamma=0.99,
        temperature=1.0,
        learning_rate=0.0001,
        n_iterations=20,  # 测试时减少迭代次数
        reg_coeff=0.01,
        verbose=True,
    )
    
    print(f"\n   训练完成!")
    print(f"   网络参数数量: {sum(p.numel() for p in network.parameters())}")
    
    # 6. 评估恢复效果
    print("\n5. 评估恢复效果")
    metrics = evaluate_deep_irl_recovery(env, recovered_reward, true_reward)
    
    print(f"   均方误差 (MSE): {metrics['mse']:.4f}")
    print(f"   平均绝对误差 (MAE): {metrics['mae']:.4f}")
    print(f"   皮尔逊相关性: {metrics['correlation']:.4f}")
    print(f"   余弦相似度: {metrics['cosine_similarity']:.4f}")
    
    print(f"\n   真实奖励统计:")
    print(f"     范围: [{metrics['true_stats']['min']:.2f}, {metrics['true_stats']['max']:.2f}]")
    print(f"     均值: {metrics['true_stats']['mean']:.2f}, 标准差: {metrics['true_stats']['std']:.2f}")
    
    print(f"   恢复奖励统计:")
    print(f"     范围: [{metrics['recovered_stats']['min']:.2f}, {metrics['recovered_stats']['max']:.2f}]")
    print(f"     均值: {metrics['recovered_stats']['mean']:.2f}, 标准差: {metrics['recovered_stats']['std']:.2f}")
    
    # 7. 可视化结果
    print("\n6. 可视化结果")
    visualize_deep_irl_results(
        env=env,
        true_reward=true_reward,
        recovered_reward=recovered_reward,
        losses=losses,
        title="Deep Maximum Entropy IRL: Test Results",
        save_path=None,  # 不保存文件
    )
    
    # 8. 保存模型
    print("\n7. 保存训练好的模型")
    torch.save({
        'network_state_dict': network.state_dict(),
        'recovered_reward': recovered_reward,
        'losses': losses,
        'metrics': metrics,
    }, "deep_maxent_irl_model.pth")
    print("   模型已保存到: deep_maxent_irl_model.pth")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_deep_maximum_entropy_irl()