"""
策略训练模块（Student4 任务）
使用恢复的奖励函数训练策略，并与真实奖励训练的策略进行比较
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from environment import AdvancedGridWorld


def value_iteration_with_reward(
    env: AdvancedGridWorld,
    reward_matrix: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-4,
    max_iter: int = 2000,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用给定的奖励矩阵进行价值迭代
    
    :param env: AdvancedGridWorld 环境
    :param reward_matrix: 奖励矩阵 (grid_size, grid_size) 或 (n_states,)
    :param gamma: 折扣因子
    :param theta: 收敛阈值
    :param max_iter: 最大迭代次数
    :param verbose: 是否打印收敛信息
    :return: 最优策略 (n_states, n_actions)，最优价值函数 (n_states,)
    """
    n_states = env.n_states
    n_actions = env.n_actions
    P = env.transition_matrix
    
    # 确保奖励是一维向量
    if reward_matrix.ndim == 2:
        R = reward_matrix.flatten()
    else:
        R = reward_matrix
    assert len(R) == n_states, f"奖励向量长度 {len(R)} 不等于状态数 {n_states}"
    
    # 初始化价值函数
    V = np.zeros(n_states)
    
    # 价值迭代主循环
    for i in range(max_iter):
        delta = 0
        # 对每个状态更新价值
        for s in range(n_states):
            old_v = V[s]
            # 计算每个动作的Q值
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = np.sum(P[s, a] * (R + gamma * V))
            # 最优价值是最大Q值
            V[s] = np.max(q_values)
            delta = max(delta, abs(old_v - V[s]))
        # 收敛判断
        if delta < theta:
            if verbose:
                print(f"价值迭代在第 {i + 1} 轮收敛，delta={delta:.6f}")
            break
    else:
        if verbose:
            print(f"价值迭代在 {max_iter} 轮后未收敛，最终 delta={delta:.6f}")
    
    # 提取最优策略
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            q_values[a] = np.sum(P[s, a] * (R + gamma * V))
        # 最优动作：Q值最大的动作
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0  # 确定性策略
    
    return policy, V


def evaluate_policy(
    env: AdvancedGridWorld,
    policy: np.ndarray,
    n_episodes: int = 100,
    max_steps: int = 100,
    gamma: float = 0.99,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    评估策略在环境中的性能
    
    :param env: 环境
    :param policy: 策略矩阵 (n_states, n_actions)
    :param n_episodes: 评估的回合数
    :param max_steps: 每个回合的最大步数
    :param gamma: 折扣因子（用于计算折扣回报）
    :param verbose: 是否打印评估信息
    :return: 包含评估指标的字典
    """
    total_returns = []
    discounted_returns = []
    episode_lengths = []
    success_count = 0
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        discounted_return = 0.0
        steps = 0
        discount = 1.0
        
        while not done and steps < max_steps:
            # 根据策略选择动作（确定性策略）
            # 确保state是整数索引
            if isinstance(state, tuple):
                state_idx = env._state_to_idx(state)
            else:
                state_idx = state
            action = np.argmax(policy[state_idx])
            next_state, reward, done, _ = env.step(action)
            
            episode_return += reward
            discounted_return += discount * reward
            discount *= gamma
            
            state = next_state
            steps += 1
        
        total_returns.append(episode_return)
        discounted_returns.append(discounted_return)
        episode_lengths.append(steps)
        
        # 检查是否成功到达终点
        # 确保state是整数索引
        if isinstance(state, tuple):
            state_coord = state
        else:
            state_coord = env._idx_to_state(state)
        if done and state_coord == env.goal_state:
            success_count += 1
    
    metrics = {
        "mean_return": float(np.mean(total_returns)),
        "std_return": float(np.std(total_returns)),
        "mean_discounted_return": float(np.mean(discounted_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": success_count / n_episodes,
        "max_return": float(np.max(total_returns)),
        "min_return": float(np.min(total_returns)),
    }
    
    if verbose:
        print(f"策略评估结果 ({n_episodes} 回合):")
        print(f"  平均回报: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
        print(f"  平均折扣回报: {metrics['mean_discounted_return']:.2f}")
        print(f"  平均轨迹长度: {metrics['mean_length']:.2f}")
        print(f"  成功率: {metrics['success_rate']:.2%}")
        print(f"  最大回报: {metrics['max_return']:.2f}")
        print(f"  最小回报: {metrics['min_return']:.2f}")
    
    return metrics


def train_policy_with_recovered_reward(
    env: AdvancedGridWorld,
    recovered_reward: np.ndarray,
    gamma: float = 0.99,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    使用恢复的奖励函数训练策略
    
    :param env: 环境
    :param recovered_reward: 恢复的奖励矩阵 (grid_size, grid_size)
    :param gamma: 折扣因子
    :param verbose: 是否打印训练信息
    :return: 训练的策略，价值函数，以及训练信息
    """
    if verbose:
        print("使用恢复的奖励函数训练策略...")
    
    # 使用恢复的奖励进行价值迭代
    policy, V = value_iteration_with_reward(
        env, recovered_reward, gamma=gamma, verbose=verbose
    )
    
    # 评估策略
    metrics = evaluate_policy(env, policy, n_episodes=50, verbose=verbose)
    
    return policy, V, metrics


def compare_policies(
    env: AdvancedGridWorld,
    true_reward: np.ndarray,
    recovered_rewards: Dict[str, np.ndarray],
    gamma: float = 0.99,
    n_episodes: int = 200,
) -> Dict[str, Dict]:
    """
    比较使用真实奖励和恢复奖励训练的策略性能
    
    :param env: 环境
    :param true_reward: 真实奖励矩阵
    :param recovered_rewards: 字典，键为算法名称，值为恢复的奖励矩阵
    :param gamma: 折扣因子
    :param n_episodes: 评估每个策略的回合数
    :return: 包含所有策略性能指标的字典
    """
    results = {}
    
    # 使用真实奖励训练策略
    print("=== 使用真实奖励训练策略 ===")
    true_policy, true_V = value_iteration_with_reward(
        env, true_reward, gamma=gamma, verbose=True
    )
    true_metrics = evaluate_policy(
        env, true_policy, n_episodes=n_episodes, verbose=True
    )
    results["ground_truth"] = {
        "policy": true_policy,
        "value_function": true_V,
        "metrics": true_metrics,
    }
    
    # 使用每个恢复的奖励训练策略
    for algo_name, recovered_reward in recovered_rewards.items():
        print(f"\n=== 使用 {algo_name} 恢复的奖励训练策略 ===")
        try:
            policy, V, metrics = train_policy_with_recovered_reward(
                env, recovered_reward, gamma=gamma, verbose=True
            )
            results[algo_name] = {
                "policy": policy,
                "value_function": V,
                "metrics": metrics,
                "reward": recovered_reward,
            }
        except Exception as e:
            print(f"算法 {algo_name} 失败: {e}")
            results[algo_name] = {
                "error": str(e),
                "policy": None,
                "value_function": None,
                "metrics": None,
            }
    
    # 打印比较结果
    print("\n" + "=" * 60)
    print("策略性能比较结果")
    print("=" * 60)
    print(f"{'算法':<20} {'平均回报':<12} {'成功率':<10} {'平均步长':<12}")
    print("-" * 60)
    
    for name, data in results.items():
        if "metrics" in data and data["metrics"] is not None:
            metrics = data["metrics"]
            print(
                f"{name:<20} {metrics['mean_return']:<12.2f} "
                f"{metrics['success_rate']:<10.2%} {metrics['mean_length']:<12.2f}"
            )
        else:
            print(f"{name:<20} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    return results


def compute_policy_similarity(
    policy_a: np.ndarray,
    policy_b: np.ndarray,
) -> float:
    """
    计算两个策略之间的相似度（状态维度上动作一致的比例）
    
    :param policy_a: 策略A (n_states, n_actions)
    :param policy_b: 策略B (n_states, n_actions)
    :return: 相似度比例 [0, 1]
    """
    assert policy_a.shape == policy_b.shape
    # 确定性策略：取最大概率动作
    actions_a = np.argmax(policy_a, axis=1)
    actions_b = np.argmax(policy_b, axis=1)
    matches = np.sum(actions_a == actions_b)
    similarity = matches / policy_a.shape[0]
    return similarity


def main():
    """主函数：演示策略训练和比较"""
    import matplotlib.pyplot as plt
    from irl_algorithms import (
        linear_programming_irl,
        maximum_margin_irl,
        maximum_entropy_irl,
    )
    from environment import value_iteration
    
    print("=" * 70)
    print("策略训练与比较演示 (Student4 任务)")
    print("=" * 70)
    
    # 1. 初始化环境
    print("\n1. 初始化环境")
    env = AdvancedGridWorld()
    true_reward = env.get_true_reward()
    print(f"   网格大小: {env.grid_size}x{env.grid_size}")
    print(f"   真实奖励范围: [{true_reward.min():.2f}, {true_reward.max():.2f}]")
    
    # 2. 生成专家策略
    print("\n2. 生成专家策略 (使用真实奖励)")
    expert_policy, expert_V = value_iteration(env)
    print(f"   专家策略形状: {expert_policy.shape}")
    
    # 3. 使用不同IRL算法恢复奖励
    print("\n3. 运行IRL算法恢复奖励函数")
    recovered_rewards = {}
    
    # 线性规划IRL
    print("   - 线性规划IRL")
    try:
        weights_lp, reward_lp = linear_programming_irl(env, expert_policy)
        recovered_rewards["Linear Programming IRL"] = reward_lp
    except Exception as e:
        print(f"     失败: {e}")
    
    # 最大间隔IRL
    print("   - 最大间隔IRL")
    try:
        weights_mm, reward_mm = maximum_margin_irl(env, expert_policy)
        recovered_rewards["Maximum Margin IRL"] = reward_mm
    except Exception as e:
        print(f"     失败: {e}")
    
    # 最大熵IRL (需要专家轨迹)
    print("   - 最大熵IRL")
    try:
        expert_trajectories = env.generate_expert_dataset(
            expert_policy=expert_policy,
            n_trajectories=50,
            noise_level=0.0,
        )
        weights_me, reward_me, _ = maximum_entropy_irl(
            env, expert_trajectories, verbose=False
        )
        recovered_rewards["Maximum Entropy IRL"] = reward_me
    except Exception as e:
        print(f"     失败: {e}")
    
    if not recovered_rewards:
        print("所有IRL算法均失败，无法进行策略比较")
        return
    
    # 4. 比较策略性能
    print("\n4. 比较策略性能")
    results = compare_policies(env, true_reward, recovered_rewards, n_episodes=100)
    
    # 5. 计算策略相似度
    print("\n5. 计算策略相似度")
    true_policy = results["ground_truth"]["policy"]
    for algo_name, data in results.items():
        if algo_name == "ground_truth":
            continue
        if "policy" in data and data["policy"] is not None:
            similarity = compute_policy_similarity(true_policy, data["policy"])
            print(f"   {algo_name} 与真实策略相似度: {similarity:.2%}")
    
    # 6. 可视化
    print("\n6. 生成可视化图表")
    visualize_comparison(results, env)
    
    print("\n演示完成！")


def visualize_comparison(results: Dict, env: AdvancedGridWorld):
    """可视化比较结果"""
    import matplotlib.pyplot as plt
    
    # 提取性能指标
    algorithms = []
    mean_returns = []
    success_rates = []
    
    for name, data in results.items():
        if "metrics" in data and data["metrics"] is not None:
            algorithms.append(name)
            mean_returns.append(data["metrics"]["mean_return"])
            success_rates.append(data["metrics"]["success_rate"])
    
    if len(algorithms) < 2:
        print("   数据不足，无法生成可视化")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均回报条形图
    axes[0, 0].bar(algorithms, mean_returns, color='skyblue')
    axes[0, 0].set_title("平均回报比较", fontsize=14)
    axes[0, 0].set_ylabel("平均回报", fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. 成功率条形图
    axes[0, 1].bar(algorithms, success_rates, color='lightgreen')
    axes[0, 1].set_title("成功率比较", fontsize=14)
    axes[0, 1].set_ylabel("成功率", fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. 策略价值函数热图（仅显示前两个算法）
    if "ground_truth" in results and results["ground_truth"]["value_function"] is not None:
        true_V = results["ground_truth"]["value_function"].reshape((env.grid_size, env.grid_size))
        im0 = axes[1, 0].imshow(true_V, cmap="viridis", origin="upper")
        axes[1, 0].set_title("真实奖励策略价值函数", fontsize=12)
        axes[1, 0].set_xlabel("Y坐标", fontsize=10)
        axes[1, 0].set_ylabel("X坐标", fontsize=10)
        plt.colorbar(im0, ax=axes[1, 0], label="价值")
    
    # 显示一个恢复奖励策略的价值函数
    recovered_algo = None
    for name in algorithms:
        if name != "ground_truth" and name in results and results[name]["value_function"] is not None:
            recovered_algo = name
            break
    
    if recovered_algo:
        rec_V = results[recovered_algo]["value_function"].reshape((env.grid_size, env.grid_size))
        im1 = axes[1, 1].imshow(rec_V, cmap="viridis", origin="upper")
        axes[1, 1].set_title(f"{recovered_algo}策略价值函数", fontsize=12)
        axes[1, 1].set_xlabel("Y坐标", fontsize=10)
        axes[1, 1].set_ylabel("X坐标", fontsize=10)
        plt.colorbar(im1, ax=axes[1, 1], label="价值")
    
    plt.suptitle("策略训练比较分析", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()