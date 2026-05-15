"""
逆强化学习算法实现（Student2 任务）
包含：线性规划IRL、最大间隔IRL、最大熵IRL、偏好IRL、混合IRL
"""

import numpy as np
import cvxpy as cp
from scipy.special import expit
from typing import Tuple, Optional, Dict, List


# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
def normalize(arr):
    """Min-max归一化"""
    arr = np.nan_to_num(arr)
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val + 1e-8)


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

    P_pi = np.einsum('sa,san->sn', policy, env.transition_matrix)

    if initial_dist is None:
        mu0 = np.zeros(n_states)
        mu0[env._state_to_idx(env.start_state)] = 1.0
    else:
        mu0 = initial_dist

    A = np.eye(n_states) - gamma * P_pi.T
    try:
        d = np.linalg.solve(A, mu0)
    except np.linalg.LinAlgError:
        d = np.linalg.pinv(A) @ mu0

    return d @ env.feature_matrix


def get_expert_initial_distribution(env, expert_trajectories):
    """从专家轨迹中估计初始状态分布"""
    n = env.n_states
    init_dist = np.zeros(n)
    for traj in expert_trajectories:
        s0 = traj["states"][0]
        idx = env._state_to_idx(s0)
        init_dist[idx] += 1
    return init_dist / (len(expert_trajectories) + 1e-8)


# ------------------------------------------------------------------------------
# 1. 线性规划IRL (Ng & Russell, 2000)
# ------------------------------------------------------------------------------
def linear_programming_irl(
    env,
    expert_policy: np.ndarray,
    gamma: float = 0.99,
    margin: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    线性规划IRL (Ng & Russell, 2000)
    通过线性规划求解奖励权重，使得专家策略最优

    :param env: AdvancedGridWorld 环境
    :param expert_policy: 专家策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param margin: 间隔大小
    :return: 奖励权重 w (n_features,)，恢复的奖励矩阵 (grid_size, grid_size)
    """
    n_states, n_actions = env.n_states, env.n_actions
    Phi = env.feature_matrix
    expert_act = np.argmax(expert_policy, axis=1)

    w = cp.Variable(Phi.shape[1])
    r = Phi @ w
    objective = cp.Minimize(cp.norm(w, 2))
    constraints = []

    for s in range(n_states):
        a_e = expert_act[s]
        for a in range(n_actions):
            if a == a_e:
                continue
            t_diff = env.transition_matrix[s, a_e] - env.transition_matrix[s, a]
            constraints.append(r[s] >= gamma * t_diff @ Phi @ w + margin)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-5, max_iters=3000)
    except cp.SolverError:
        prob.solve(solver=cp.ECOS, max_iters=3000)

    weights = w.value if w.value is not None else np.ones(Phi.shape[1])
    reward = (Phi @ weights).reshape(env.grid_size, env.grid_size)
    return weights, reward


def linear_programming_irl_ng_russell(
    env,
    expert_policy: np.ndarray,
    gamma: float = 0.99,
    margin: float = 0.015,
    reward_bound: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    线性规划IRL (Ng & Russell, 2000) — 向后兼容包装

    :param env: AdvancedGridWorld 环境
    :param expert_policy: 专家策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param margin: 间隔大小
    :param reward_bound: 奖励绝对值上限（当前实现中通过范数正则化约束）
    :return: 奖励权重 w (n_features,)，恢复的奖励矩阵 (grid_size, grid_size)
    """
    return linear_programming_irl(env, expert_policy, gamma=gamma, margin=margin)


# ------------------------------------------------------------------------------
# 2. 最大间隔IRL (Abbeel & Ng, 2004)
# ------------------------------------------------------------------------------
def maximum_margin_irl(
    env,
    expert_policy: np.ndarray,
    gamma: float = 0.99,
    margin: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    最大间隔IRL (Abbeel & Ng, 2004)
    通过Ng-Russell约束形式最大化专家策略与所有其他动作之间的间隔

    :param env: AdvancedGridWorld 环境
    :param expert_policy: 专家策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param margin: 最小间隔要求
    :return: 奖励权重 w (n_features,)，恢复的奖励矩阵 (grid_size, grid_size)
    """
    n_states, n_actions = env.n_states, env.n_actions
    Phi = env.feature_matrix
    expert_act = np.argmax(expert_policy, axis=1)
    w = cp.Variable(Phi.shape[1])
    r = Phi @ w
    xi = cp.Variable(n_states * n_actions)
    idx = 0

    objective = cp.Minimize(cp.norm(w, 2) + 0.1 * cp.sum(xi))
    constraints = [xi >= 0]

    for s in range(n_states):
        a_e = expert_act[s]
        for a in range(n_actions):
            if a == a_e:
                continue
            t_diff = env.transition_matrix[s, a_e] - env.transition_matrix[s, a]
            constraints.append(r[s] >= gamma * t_diff @ Phi @ w + margin - xi[idx])
            idx += 1

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-5, max_iters=3000)
    except cp.SolverError:
        prob.solve(solver=cp.ECOS, max_iters=3000)

    weights = w.value if w.value is not None else np.ones(Phi.shape[1])
    reward = (Phi @ weights).reshape(env.grid_size, env.grid_size)
    return weights, reward


# ------------------------------------------------------------------------------
# 最大熵IRL 辅助函数
# ------------------------------------------------------------------------------
def compute_expert_feature_expectations_from_trajectories(
    env,
    trajectories: List[Dict],
    gamma: float = 0.99,
) -> np.ndarray:
    """
    从专家轨迹计算特征期望（含无限尾部折扣）
    :param env: AdvancedGridWorld环境
    :param trajectories: 轨迹列表，每个轨迹包含states, actions, rewards等
    :param gamma: 折扣因子
    :return: 特征期望向量 (n_features,)
    """
    n_features = env.feature_matrix.shape[1]
    mu_expert = np.zeros(n_features)

    for traj in trajectories:
        discount = 1.0
        for s in traj["states"]:
            idx = env._state_to_idx(s)
            mu_expert += discount * env.feature_matrix[idx]
            discount *= gamma
        # 无限尾部折扣：假设从最后状态继续执行
        last_state_idx = env._state_to_idx(traj["states"][-1])
        infinite_tail_discount = discount / (1 - gamma)
        mu_expert += infinite_tail_discount * env.feature_matrix[last_state_idx]

    mu_expert /= len(trajectories)
    return mu_expert


def soft_value_iteration(
    env,
    reward_weights: np.ndarray,
    gamma: float = 0.99,
    temperature: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft值迭代算法（最大熵IRL核心）
    计算给定奖励权重下的soft Q值和soft value函数

    :param env: AdvancedGridWorld环境
    :param reward_weights: 奖励权重向量 (n_features,)
    :param gamma: 折扣因子
    :param temperature: 温度参数（控制随机性）
    :param max_iter: 最大迭代次数
    :param tol: 收敛容忍度
    :return: soft Q值矩阵 (n_states, n_actions), soft value函数 (n_states,)
    """
    n_states = env.n_states
    R = np.clip(env.feature_matrix @ reward_weights, -10, 10)
    V = np.zeros(n_states)
    P = env.transition_matrix

    for _ in range(max_iter):
        V_prev = V.copy()
        expected_V = P @ V_prev
        Q = R[:, None] + gamma * expected_V

        Q_max = np.max(Q, axis=1, keepdims=True)
        V_new = Q_max + temperature * np.log(
            np.exp((Q - Q_max) / temperature).sum(axis=1, keepdims=True)
        )
        V_new = V_new.flatten()

        if np.max(np.abs(V_new - V_prev)) < tol:
            break
        V = V_new

    Q_final = R[:, None] + gamma * (P @ V)
    return Q_final, V


def compute_soft_policy(
    Q: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    从soft Q值计算softmax策略
    :param Q: soft Q值矩阵 (n_states, n_actions)
    :param temperature: 温度参数
    :return: 策略矩阵 (n_states, n_actions)
    """
    if temperature > 0:
        Q_max = np.max(Q, axis=1, keepdims=True)
        exp_q = np.exp((Q - Q_max) / temperature)
        policy = exp_q / exp_q.sum(axis=1, keepdims=True)
    else:
        policy = np.zeros_like(Q)
        best_actions = np.argmax(Q, axis=1)
        policy[np.arange(len(best_actions)), best_actions] = 1.0
    return policy


def compute_state_visitation_frequency_maxent(
    env,
    policy: np.ndarray,
    gamma: float = 0.99,
    initial_dist: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    通过解析法计算状态访问频率（最大熵IRL专用）
    :param env: AdvancedGridWorld环境
    :param policy: 策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param initial_dist: 初始状态分布
    :param max_iter: 最大迭代次数（解析失败时的回退）
    :param tol: 收敛容忍度
    :return: 状态访问频率 (n_states,)
    """
    n_states = env.n_states
    P_pi = np.einsum('sa,san->sn', policy, env.transition_matrix)

    if initial_dist is None:
        D = np.zeros(n_states)
        D[env._state_to_idx(env.start_state)] = 1.0
    else:
        D = initial_dist.copy()

    A = np.eye(n_states) - gamma * P_pi.T
    try:
        return np.linalg.solve(A, D)
    except np.linalg.LinAlgError:
        D_final = D.copy()
        for _ in range(max_iter):
            D_new = D + gamma * P_pi.T @ D_final
            if np.max(np.abs(D_new - D_final)) < tol:
                break
            D_final = D_new
        return D_final


def compute_policy_feature_expectations_maxent(
    env,
    policy: np.ndarray,
    gamma: float = 0.99,
    initial_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算策略的特征期望（最大熵IRL专用）
    :param env: AdvancedGridWorld环境
    :param policy: 策略矩阵 (n_states, n_actions)
    :param gamma: 折扣因子
    :param initial_dist: 初始状态分布
    :return: 特征期望向量 (n_features,)
    """
    D = compute_state_visitation_frequency_maxent(env, policy, gamma, initial_dist)
    return D @ env.feature_matrix


# ------------------------------------------------------------------------------
# 3. 最大熵IRL (Ziebart et al., 2008) — Adam优化 + 早停
# ------------------------------------------------------------------------------
def maximum_entropy_irl(
    env,
    expert_trajectories: List[Dict],
    gamma: float = 0.99,
    temperature: float = 0.8,
    learning_rate: float = 0.03,
    n_iterations: int = 3000,
    reg_coeff: float = 1e-4,
    verbose: bool = True,
    patience: int = 300,
    min_delta: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    最大熵IRL主算法（Ziebart et al., 2008）

    :param env: AdvancedGridWorld环境
    :param expert_trajectories: 专家轨迹列表
    :param gamma: 折扣因子
    :param temperature: 温度参数
    :param learning_rate: 学习率
    :param n_iterations: 最大迭代次数
    :param reg_coeff: 正则化系数（L2正则化）
    :param verbose: 是否打印训练信息
    :param patience: 早停等待轮数
    :param min_delta: 早停最小改进量
    :return: 奖励权重, 恢复的奖励矩阵, 损失历史
    """
    n_states = env.n_states
    n_features = env.feature_matrix.shape[1]
    Phi = env.feature_matrix

    # 1. 计算专家特征期望（含无限尾部折扣）
    mu_expert = compute_expert_feature_expectations_from_trajectories(
        env, expert_trajectories, gamma
    )

    # 2. 估计专家初始状态分布
    expert_init_dist = get_expert_initial_distribution(env, expert_trajectories)

    # 3. 初始化权重（小随机值）
    np.random.seed(42)
    weights = 1e-4 * np.random.randn(n_features)

    # 4. Adam优化器状态
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    losses = []
    best_loss = float('inf')
    best_weights = weights.copy()
    patience_counter = 0

    for it in range(1, n_iterations + 1):
        # 4.1 计算当前奖励向量
        reward_vec = Phi @ weights

        # 4.2 Soft值迭代 → softmax策略
        Q, V = soft_value_iteration(env, weights, gamma=gamma, temperature=temperature,
                                     max_iter=300, tol=1e-4)

        # 4.3 计算当前策略的状态占用
        P_pi = np.einsum('sa,san->sn', compute_soft_policy(Q, temperature), env.transition_matrix)
        A_mat = np.eye(n_states) - gamma * P_pi.T
        try:
            occ = np.linalg.solve(A_mat, expert_init_dist)
        except np.linalg.LinAlgError:
            occ = np.linalg.pinv(A_mat) @ expert_init_dist

        mu_policy = occ @ Phi

        # 4.4 Adam梯度更新
        grad = mu_expert - mu_policy - reg_coeff * weights

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** it)
        v_hat = v / (1 - beta2 ** it)

        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        # 4.5 记录损失
        loss_val = float(np.linalg.norm(mu_expert - mu_policy))
        losses.append(loss_val)

        # 4.6 早停检查
        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"[MaxEnt] 早停于第 {it} 轮 | 最佳损失: {best_loss:.4f}")
            weights = best_weights
            break

        if verbose and (it % 50 == 0 or it == n_iterations):
            gnorm = np.linalg.norm(grad)
            print(f"第 {it:4d} 轮 | 损失: {loss_val:.6f} | 梯度范数: {gnorm:.6f}")

    # 5. 计算恢复的奖励矩阵
    reward_matrix = (Phi @ weights).reshape(env.grid_size, env.grid_size)
    return weights, reward_matrix, losses


# ------------------------------------------------------------------------------
# 4. 偏好IRL (Bradley-Terry)
# ------------------------------------------------------------------------------
def preference_irl_bt(
    env,
    preferences: List[Tuple[Dict, Dict, float]],
    gamma: float = 0.99,
    lr: float = 0.05,
    reg_coeff: float = 1e-3,
    n_iterations: int = 1000,
    verbose: bool = True,
):
    """
    基于偏好的IRL (Bradley-Terry模型)
    从成对比较中学习奖励函数

    :param env: AdvancedGridWorld环境
    :param preferences: 偏好数据列表 [(traj_i, traj_j, label), ...]，label=1表示i优于j
    :param gamma: 折扣因子
    :param lr: 学习率
    :param reg_coeff: 正则化系数
    :param n_iterations: 迭代次数
    :param verbose: 是否打印训练信息
    :return: 奖励权重, 恢复的奖励矩阵, 损失历史
    """
    n_features = env.feature_matrix.shape[1]
    f_mat = env.feature_matrix

    def get_discounted_feat(traj):
        feat = np.zeros(n_features)
        disc = 1.0
        for s in traj["states"]:
            idx = env._state_to_idx(s)
            feat += disc * f_mat[idx]
            disc *= gamma
        return feat

    data = []
    for ti, tj, label in preferences:
        fi = get_discounted_feat(ti)
        fj = get_discounted_feat(tj)
        data.append((fi, fj, label))

    w = np.random.randn(n_features) * 0.01
    losses = []

    for it in range(n_iterations):
        grad = np.zeros(n_features)
        nll = 0.0

        for fi, fj, label in data:
            ri = w @ fi
            rj = w @ fj
            p = expit(ri - rj)
            p = np.clip(p, 1e-9, 1 - 1e-9)

            if label > 0.5:
                grad += (1.0 - p) * (fi - fj)
                nll -= np.log(p)
            else:
                grad -= p * (fi - fj)
                nll -= np.log(1.0 - p)

        grad = grad / len(data) - reg_coeff * w
        w += lr * grad
        losses.append(nll)

        if verbose and it % 50 == 0:
            print(f"第 {it:3d} 轮 | NLL = {nll:.4f}")

    reward = (f_mat @ w).reshape(env.grid_size, env.grid_size)
    return w, reward, losses


# ------------------------------------------------------------------------------
# 5. MaxEnt + BT 混合IRL（两阶段）
# ------------------------------------------------------------------------------
def maxent_bt_irl(
    env,
    expert_trajectories: List[Dict],
    preferences: List[Tuple[Dict, Dict, float]],
    gamma: float = 0.99,
    temperature: float = 0.8,
    lr: float = 0.01,
    bt_weight: float = 0.1,
    n_iterations: int = 2000,
    reg_coeff: float = 1e-4,
    verbose: bool = True,
):
    """
    MaxEnt + Bradley-Terry 混合IRL（两阶段训练）
    阶段1: 使用最大熵IRL从轨迹学习初始奖励
    阶段2: 使用BT偏好数据微调奖励函数

    :param env: AdvancedGridWorld环境
    :param expert_trajectories: 专家轨迹列表
    :param preferences: 偏好数据列表
    :param gamma: 折扣因子
    :param temperature: 温度参数
    :param lr: 阶段2学习率（未使用，内部使用bt_lr=0.001）
    :param bt_weight: BT阶段权重（未使用，内部使用正则化到w_maxent）
    :param n_iterations: 最大迭代次数（未使用）
    :param reg_coeff: 正则化系数
    :param verbose: 是否打印训练信息
    :return: 奖励权重, 恢复的奖励矩阵
    """
    # 阶段1: 完整MaxEnt训练
    w_maxent, _, _ = maximum_entropy_irl(
        env, expert_trajectories,
        gamma=gamma, temperature=temperature,
        learning_rate=0.03, n_iterations=3000,
        reg_coeff=1e-4, verbose=verbose
    )

    # 阶段2: BT微调
    n_features = env.feature_matrix.shape[1]
    f_mat = env.feature_matrix

    def get_discounted_feat(traj):
        feat = np.zeros(n_features)
        disc = 1.0
        for s in traj["states"]:
            idx = env._state_to_idx(s)
            feat += disc * f_mat[idx]
            disc *= gamma
        return feat

    data = []
    for ti, tj, label in preferences:
        fi = get_discounted_feat(ti)
        fj = get_discounted_feat(tj)
        data.append((fi, fj, label))

    w = w_maxent.copy()
    bt_lr = 0.001

    for it in range(800):
        grad = np.zeros(n_features)
        nll = 0.0
        for fi, fj, label in data:
            ri = w @ fi
            rj = w @ fj
            p = expit(ri - rj)
            p = np.clip(p, 1e-9, 1 - 1e-9)

            if label > 0.5:
                grad += (1.0 - p) * (fi - fj)
                nll -= np.log(p)
            else:
                grad -= p * (fi - fj)
                nll -= np.log(1.0 - p)

        grad = grad / len(data) - reg_coeff * (w - w_maxent)
        w += bt_lr * grad

        if verbose and it % 100 == 0:
            print(f"BT微调 {it:3d} | NLL = {nll:.4f}")

    final_reward = (f_mat @ w).reshape(env.grid_size, env.grid_size)
    return w, final_reward


# ------------------------------------------------------------------------------
# 评估工具
# ------------------------------------------------------------------------------
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
    assert recovered_reward.shape == true_reward.shape

    mse = np.mean((recovered_reward - true_reward) ** 2)
    mae = np.mean(np.abs(recovered_reward - true_reward))
    correlation = np.corrcoef(recovered_reward.flatten(), true_reward.flatten())[0, 1]

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

    im1 = axes[0].imshow(true_reward, cmap="RdYlGn", origin="upper")
    axes[0].set_title("Ground Truth Reward", fontsize=14)
    axes[0].set_xlabel("Y Coordinate", fontsize=12)
    axes[0].set_ylabel("X Coordinate", fontsize=12)
    plt.colorbar(im1, ax=axes[0], label="Reward Value")

    im2 = axes[1].imshow(recovered_reward, cmap="RdYlGn", origin="upper")
    axes[1].set_title("Recovered Reward", fontsize=14)
    axes[1].set_xlabel("Y Coordinate", fontsize=12)
    axes[1].set_ylabel("X Coordinate", fontsize=12)
    plt.colorbar(im2, ax=axes[1], label="Reward Value")

    for ax in axes:
        ax.text(
            env.goal_state[1], env.goal_state[0], "G",
            ha="center", va="center", color="black", fontsize=16, fontweight="bold",
        )
        ax.text(
            env.start_state[1], env.start_state[0], "S",
            ha="center", va="center", color="black", fontsize=16, fontweight="bold",
        )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ===================== 测试代码 =====================
if __name__ == "__main__":
    from environment import AdvancedGridWorld, value_iteration

    print("=== 测试IRL算法 ===")

    env = AdvancedGridWorld()
    expert_policy, expert_V = value_iteration(env)

    # 测试线性规划IRL
    print("\n--- 线性规划IRL (Ng & Russell) ---")
    try:
        weights_lp, reward_lp = linear_programming_irl(env, expert_policy)
        print(f"恢复的权重: {weights_lp}")

        true_reward = env.get_true_reward()
        metrics = evaluate_reward_recovery(env, reward_lp, true_reward)
        print(f"恢复指标: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"相关性={metrics['correlation']:.4f}")

        visualize_reward_comparison(
            env, true_reward, reward_lp, "Linear Programming IRL Recovery"
        )
    except Exception as e:
        print(f"线性规划IRL失败: {e}")

    # 测试最大间隔IRL
    print("\n--- 最大间隔IRL ---")
    try:
        weights_mm, reward_mm = maximum_margin_irl(env, expert_policy)
        print(f"恢复的权重: {weights_mm}")

        true_reward = env.get_true_reward()
        metrics = evaluate_reward_recovery(env, reward_mm, true_reward)
        print(f"恢复指标: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"相关性={metrics['correlation']:.4f}")

        visualize_reward_comparison(
            env, true_reward, reward_mm, "Maximum Margin IRL Recovery"
        )
    except Exception as e:
        print(f"最大间隔IRL失败: {e}")

    # 测试最大熵IRL
    print("\n--- 最大熵IRL (Maximum Entropy IRL) ---")
    try:
        print("生成专家轨迹数据...")
        expert_trajectories = env.generate_expert_dataset(
            expert_policy=expert_policy,
            n_trajectories=50,
            noise_level=0.0,
        )
        print(f"生成轨迹数量: {len(expert_trajectories)}")

        weights_me, reward_me, losses = maximum_entropy_irl(
            env,
            expert_trajectories,
            gamma=0.99,
            temperature=0.8,
            learning_rate=0.03,
            n_iterations=500,
            reg_coeff=1e-4,
            verbose=True,
        )
        print(f"恢复的权重: {weights_me}")

        true_reward = env.get_true_reward()
        metrics = evaluate_reward_recovery(env, reward_me, true_reward)
        print(f"恢复指标: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"相关性={metrics['correlation']:.4f}")

        visualize_reward_comparison(
            env, true_reward, reward_me, "Maximum Entropy IRL Recovery"
        )

        # 绘制损失曲线
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Maximum Entropy IRL Training Loss")
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"最大熵IRL失败: {e}")
        import traceback
        traceback.print_exc()
