"""
逆强化学习算法实现（Student2 任务）
包含：线性规划IRL、最大间隔IRL
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional, Dict, List


def compute_feature_expectations(
    env,
    policy: np.ndarray,
    gamma: float = 0.99,
    initial_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算给定策略下的折扣特征期望
    :param env: AdvancedGridWorld 环境
    :param policy: 策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param initial_dist: 初始状态分布 (n_states,)，默认为起点分布
    :return: 特征期望向量 (n_features,)
    """
    n_states = env.n_states
    n_actions = env.n_actions
    n_features = env.feature_matrix.shape[1]
    
    # 初始状态分布：如果未提供，假设起点为初始状态
    if initial_dist is None:
        mu0 = np.zeros(n_states)
        mu0[env._state_to_idx(env.start_state)] = 1.0
    else:
        mu0 = initial_dist
    
    # 构建策略转移矩阵 P_pi: (n_states, n_states)
    P_pi = np.zeros((n_states, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            prob = policy[s, a]
            if prob > 0:
                P_pi[s] += prob * env.transition_matrix[s, a]
    
    # 计算折扣状态访问分布: d = (I - gamma * P_pi)^{-1} * mu0
    # 注意：这里假设无限时间折扣，使用线性求解
    I = np.eye(n_states)
    try:
        d = np.linalg.solve(I - gamma * P_pi, mu0)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        d = np.linalg.pinv(I - gamma * P_pi) @ mu0
    
    # 特征期望: μ = d^T * Φ
    mu = d @ env.feature_matrix
    return mu


def linear_programming_irl(
    env,
    expert_policy: np.ndarray,
    gamma: float = 0.99,
    margin: float = 1.0,
    reward_norm: str = "l2",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    线性规划IRL (Ng & Russell, 2000)
    通过线性规划求解奖励权重，使得专家策略最优
    
    算法步骤：
    1. 计算专家特征期望 μ_expert
    2. 对每个随机策略（或采样策略）计算特征期望 μ_other
    3. 构建线性规划：最大化间隔，使得 w^T μ_expert ≥ w^T μ_other + margin
    4. 加入正则化防止过拟合
    
    :param env: AdvancedGridWorld 环境
    :param expert_policy: 专家策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param margin: 间隔大小
    :param reward_norm: 正则化类型 ("l1" 或 "l2")
    :return: 奖励权重 w (n_features,)，恢复的奖励矩阵 (grid_size, grid_size)
    """
    n_features = env.feature_matrix.shape[1]
    
    # 1. 计算专家特征期望
    mu_expert = compute_feature_expectations(env, expert_policy, gamma)
    
    # 2. 生成一组随机策略作为对比
    n_random_policies = 50
    random_policies = []
    for _ in range(n_random_policies):
        # 生成随机策略（每个状态的动作分布均匀随机）
        policy = np.random.rand(env.n_states, env.n_actions)
        policy = policy / policy.sum(axis=1, keepdims=True)
        random_policies.append(policy)
    
    # 3. 计算每个随机策略的特征期望
    mu_others = []
    for policy in random_policies:
        mu = compute_feature_expectations(env, policy, gamma)
        mu_others.append(mu)
    mu_others = np.array(mu_others)  # (n_random_policies, n_features)
    
    # 4. 构建线性规划问题
    w = cp.Variable(n_features)
    
    # 目标函数：最小化权重范数（防止过拟合）
    if reward_norm == "l1":
        objective = cp.Minimize(cp.norm(w, 1))
    else:  # 默认L2正则化
        objective = cp.Minimize(cp.norm(w, 2))
    
    # 约束：专家特征期望优于所有随机策略
    constraints = []
    for i in range(len(mu_others)):
        constraints.append(mu_expert @ w >= mu_others[i] @ w + margin)
    
    # 添加边界约束：权重绝对值不超过10
    constraints.append(w >= -10)
    constraints.append(w <= 10)
    
    # 5. 求解线性规划
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        # 如果ECOS失败，尝试其他求解器
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            raise ValueError("线性规划求解失败，请检查问题是否可行")
    
    if w.value is None:
        raise ValueError("线性规划无可行解")
    
    weights = w.value
    
    # 6. 计算恢复的奖励矩阵
    reward_matrix = env.feature_matrix @ weights
    reward_matrix = reward_matrix.reshape((env.grid_size, env.grid_size))
    
    return weights, reward_matrix


def linear_programming_irl_ng_russell(
    env,
    expert_policy: np.ndarray,
    gamma: float = 0.99,
    margin: float = 1.0,
    reward_bound: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    线性规划IRL (Ng & Russell, 2000) 精确实现
    
    算法公式：
    最大化 ∑_s r(s)
    约束: (P_{π_E} - P_a)(I - γ P_{π_E})^{-1} r ≥ margin, ∀s,a ≠ π_E(s)
          |r(s)| ≤ reward_bound
    
    其中 r = Φw，w 是权重向量。
    
    :param env: AdvancedGridWorld 环境
    :param expert_policy: 专家策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param margin: 间隔大小
    :param reward_bound: 奖励绝对值上限
    :return: 奖励权重 w (n_features,)，恢复的奖励矩阵 (grid_size, grid_size)
    """
    n_states = env.n_states
    n_actions = env.n_actions
    n_features = env.feature_matrix.shape[1]
    
    # 1. 提取专家动作（确定性策略）
    expert_actions = np.argmax(expert_policy, axis=1)  # (n_states,)
    
    # 2. 构建专家策略转移矩阵 P_pi
    P_pi = np.zeros((n_states, n_states))
    for s in range(n_states):
        a_expert = expert_actions[s]
        P_pi[s] = env.transition_matrix[s, a_expert]
    
    # 3. 计算矩阵 A = (I - gamma * P_pi)^{-1}
    I = np.eye(n_states)
    try:
        A = np.linalg.inv(I - gamma * P_pi)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        A = np.linalg.pinv(I - gamma * P_pi)
    
    # 4. 计算 M = A @ Φ
    M = A @ env.feature_matrix  # (n_states, n_features)
    
    # 5. 构建线性规划约束
    w = cp.Variable(n_features)
    
    # 目标函数：最大化奖励和（等价于最小化负和）
    objective = cp.Maximize(cp.sum(env.feature_matrix @ w))
    
    constraints = []
    
    # 对于每个状态s和每个非专家动作a，添加间隔约束
    for s in range(n_states):
        a_expert = expert_actions[s]
        for a in range(n_actions):
            if a == a_expert:
                continue
            # 计算差值向量 d = P[s, a_expert, :] - P[s, a, :]
            d = env.transition_matrix[s, a_expert] - env.transition_matrix[s, a]
            # 约束: d @ M @ w >= margin
            constraints.append(d @ M @ w >= margin)
    
    # 奖励边界约束：|φ(s)^T w| <= reward_bound
    for s in range(n_states):
        phi = env.feature_matrix[s]
        constraints.append(phi @ w <= reward_bound)
        constraints.append(phi @ w >= -reward_bound)
    
    # 6. 求解线性规划
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            raise ValueError("线性规划求解失败")
    
    if w.value is None:
        raise ValueError("线性规划无可行解")
    
    weights = w.value
    
    # 7. 计算恢复的奖励矩阵
    reward_matrix = env.feature_matrix @ weights
    reward_matrix = reward_matrix.reshape((env.grid_size, env.grid_size))
    
    return weights, reward_matrix


def maximum_margin_irl(
    env,
    expert_policy: np.ndarray,
    gamma: float = 0.99,
    margin: float = 1.0,
    reward_norm: str = "l2",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    最大间隔IRL (Abbeel & Ng, 2004)
    最大化专家策略与所有其他策略之间的特征期望间隔
    
    算法步骤：
    1. 计算专家特征期望 μ_expert
    2. 使用线性规划最大化间隔
    3. 加入正则化防止过拟合
    
    :param env: AdvancedGridWorld 环境
    :param expert_policy: 专家策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param margin: 最小间隔要求
    :param reward_norm: 正则化类型 ("l1" 或 "l2")
    :return: 奖励权重 w (n_features,)，恢复的奖励矩阵 (grid_size, grid_size)
    """
    n_features = env.feature_matrix.shape[1]
    
    # 1. 计算专家特征期望
    mu_expert = compute_feature_expectations(env, expert_policy, gamma)
    
    # 2. 生成一组随机策略作为对比
    n_random_policies = 100
    random_policies = []
    for _ in range(n_random_policies):
        # 生成随机策略
        policy = np.random.rand(env.n_states, env.n_actions)
        policy = policy / policy.sum(axis=1, keepdims=True)
        random_policies.append(policy)
    
    # 3. 计算每个随机策略的特征期望
    mu_others = []
    for policy in random_policies:
        mu = compute_feature_expectations(env, policy, gamma)
        mu_others.append(mu)
    mu_others = np.array(mu_others)  # (n_random_policies, n_features)
    
    # 4. 构建最大间隔优化问题
    w = cp.Variable(n_features)
    xi = cp.Variable(len(mu_others))  # 松弛变量
    
    # 目标函数：最小化权重范数 + 松弛惩罚
    if reward_norm == "l1":
        reg_term = cp.norm(w, 1)
    else:
        reg_term = cp.norm(w, 2)
    
    # 添加松弛变量惩罚
    C = 1.0  # 惩罚系数
    objective = cp.Minimize(reg_term + C * cp.sum(xi))
    
    # 约束：专家特征期望优于其他策略至少margin - xi
    constraints = [xi >= 0]
    for i in range(len(mu_others)):
        constraints.append(mu_expert @ w >= mu_others[i] @ w + margin - xi[i])
    
    # 权重边界约束
    constraints.append(w >= -10)
    constraints.append(w <= 10)
    
    # 5. 求解优化问题
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            raise ValueError("最大间隔优化求解失败")
    
    if w.value is None:
        raise ValueError("最大间隔优化无可行解")
    
    weights = w.value
    
    # 6. 计算恢复的奖励矩阵
    reward_matrix = env.feature_matrix @ weights
    reward_matrix = reward_matrix.reshape((env.grid_size, env.grid_size))
    
    return weights, reward_matrix


def evaluate_reward_recovery(
    env,
    recovered_reward: np.ndarray,
    true_reward: np.ndarray,
) -> Dict[str, float]:
    """
    评估恢复的奖励与真实奖励之间的差异
    :param recovered_reward: 恢复的奖励矩阵 (grid_size, grid_size)
    :param true_reward: 真实奖励矩阵 (grid_size, grid_size)
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
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        "cosine_similarity": cosine_sim,
    }


def visualize_reward_comparison(
    env,
    true_reward: np.ndarray,
    recovered_reward: np.ndarray,
    title: str = "Reward Recovery Comparison",
    save_path: Optional[str] = None,
):
    """
    可视化真实奖励与恢复奖励的对比
    :param env: 环境对象（用于绘图）
    :param true_reward: 真实奖励矩阵
    :param recovered_reward: 恢复的奖励矩阵
    :param title: 图标题
    :param save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 真实奖励热力图
    im1 = axes[0].imshow(true_reward, cmap="RdYlGn", origin="upper")
    axes[0].set_title("Ground Truth Reward", fontsize=14)
    axes[0].set_xlabel("Y Coordinate", fontsize=12)
    axes[0].set_ylabel("X Coordinate", fontsize=12)
    plt.colorbar(im1, ax=axes[0], label="Reward Value")
    
    # 恢复奖励热力图
    im2 = axes[1].imshow(recovered_reward, cmap="RdYlGn", origin="upper")
    axes[1].set_title("Recovered Reward", fontsize=14)
    axes[1].set_xlabel("Y Coordinate", fontsize=12)
    axes[1].set_ylabel("X Coordinate", fontsize=12)
    plt.colorbar(im2, ax=axes[1], label="Reward Value")
    
    # 标注起点和终点
    for ax in axes:
        ax.text(
            env.goal_state[1],
            env.goal_state[0],
            "G",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )
        ax.text(
            env.start_state[1],
            env.start_state[0],
            "S",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 示例用法
    from environment import AdvancedGridWorld, value_iteration
    
    print("=== 测试IRL算法 ===")
    
    # 初始化环境
    env = AdvancedGridWorld()
    
    # 生成专家最优策略
    expert_policy, expert_V = value_iteration(env)
    
    # 测试线性规划IRL（基础版本）
    print("\n--- 线性规划IRL（基础版本）---")
    try:
        weights_lp, reward_lp = linear_programming_irl(env, expert_policy)
        print(f"恢复的权重: {weights_lp}")
        
        # 评估恢复效果
        true_reward = env.get_true_reward()
        metrics = evaluate_reward_recovery(env, reward_lp, true_reward)
        print(f"恢复指标: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"相关性={metrics['correlation']:.4f}")
        
        # 可视化对比
        visualize_reward_comparison(
            env, true_reward, reward_lp, "Linear Programming IRL (Basic) Recovery"
        )
    except Exception as e:
        print(f"线性规划IRL失败: {e}")
    
    # 测试线性规划IRL（Ng & Russell 精确版本）
    print("\n--- 线性规划IRL（Ng & Russell 精确版本）---")
    try:
        weights_lp2, reward_lp2 = linear_programming_irl_ng_russell(env, expert_policy, margin=1.0, reward_bound=10.0)
        print(f"恢复的权重: {weights_lp2}")
        
        # 评估恢复效果
        true_reward = env.get_true_reward()
        metrics = evaluate_reward_recovery(env, reward_lp2, true_reward)
        print(f"恢复指标: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"相关性={metrics['correlation']:.4f}")
        
        # 可视化对比
        visualize_reward_comparison(
            env, true_reward, reward_lp2, "Linear Programming IRL (Ng & Russell) Recovery"
        )
    except Exception as e:
        print(f"线性规划IRL (Ng & Russell) 失败: {e}")
    
    # 测试最大间隔IRL
    print("\n--- 最大间隔IRL ---")
    try:
        weights_mm, reward_mm = maximum_margin_irl(env, expert_policy)
        print(f"恢复的权重: {weights_mm}")
        
        # 评估恢复效果
        true_reward = env.get_true_reward()
        metrics = evaluate_reward_recovery(env, reward_mm, true_reward)
        print(f"恢复指标: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"相关性={metrics['correlation']:.4f}")
        
        # 可视化对比
        visualize_reward_comparison(
            env, true_reward, reward_mm, "Maximum Margin IRL Recovery"
        )
    except Exception as e:
        print(f"最大间隔IRL失败: {e}")
