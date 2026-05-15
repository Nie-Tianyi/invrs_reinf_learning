# compute_gt.py  单独计算 Ground Truth 指标
import numpy as np
from environment import AdvancedGridWorld

# ===================== 复制你原版的函数（完全一致） =====================
def get_expert_policy(env):
    R = env.true_reward_matrix.flatten()
    P = env.transition_matrix
    gamma = 0.99
    V = np.zeros(env.n_states)
    for _ in range(1000):
        V_new = np.max(np.sum(P * (R + gamma * V), axis=2), axis=1)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        best_a = np.argmax(np.sum(P[s] * (R + gamma * V), axis=1))
        policy[s, best_a] = 1.0
    return policy

def evaluate_policy_normal(env, policy, episodes=200):
    success = 0
    total_r = 0.0
    total_steps = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        steps = 0
        while not done and steps < 100:
            idx = env._state_to_idx(s)
            a = np.argmax(policy[idx])
            s, r, done, _ = env.step(a)
            ep_r += r
            steps += 1
        total_r += ep_r
        total_steps += steps
        if done and env.is_terminal(s):
            success += 1
    return {
        "success_rate": success / episodes,
        "mean_reward": total_r / episodes,
        "mean_steps": total_steps / episodes
    }

# ===================== 单独运行 GT 评估 =====================
if __name__ == "__main__":
    # 和你实验参数完全一致
    seed = 42
    np.random.seed(seed)
    
    env = AdvancedGridWorld({
        "grid_size": 10,
        "max_steps": 500
    })
    
    pi_e = get_expert_policy(env)
    gt_result = evaluate_policy_normal(env, pi_e)
    
    print("\n========== Ground Truth 单独输出 ==========")
    print(f"Mean Reward: {gt_result['mean_reward']:.2f}")
    print(f"Mean Steps:  {gt_result['mean_steps']:.2f}")
    print(f"Success Rate: {gt_result['success_rate']:.1%}")
    print("===========================================\n")
