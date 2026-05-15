import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap

from scipy.special import logsumexp, expit

# ===================== 颜色 =====================
MY_COLORS = ["#eaf3e2", "#b4deb6", "#7bc6be", "#439cc4", "#0868a6"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom", MY_COLORS)

# ===================== 导入 =====================
from environment import AdvancedGridWorld
from irl_algorithms import (
    linear_programming_irl,
    maximum_margin_irl,
    maximum_entropy_irl,
    preference_irl_bt,
    maxent_bt_irl
)

# ===================== 路径 =====================
os.makedirs("result", exist_ok=True)
os.makedirs("checkpoint", exist_ok=True)
CHECKPOINT_PATH = "checkpoint/ablation_final.pkl"

# ===========================================================================
def value_iteration_with_reward(env, reward_vec, gamma=0.99, theta=1e-6, max_iter=1000):
    n_states = env.n_states
    n_actions = env.n_actions
    P = env.transition_matrix
    R = reward_vec

    V = np.zeros(n_states)
    for _ in range(max_iter):
        delta = 0
        for s in range(n_states):
            old_v = V[s]
            q_vals = np.sum(P[s] * (R + gamma * V), axis=1)
            V[s] = np.max(q_vals)
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < theta:
            break
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        best_a = np.argmax(np.sum(P[s] * (R + gamma * V), axis=1))
        policy[s, best_a] = 1.0
    return policy, V

# ===========================================================================
def get_policy_from_maxent_weights(env, weights, temperature=0.75):
    reward_vec = env.feature_matrix @ weights
    R = np.clip(reward_vec, -20, 20)
    P = env.transition_matrix
    gamma = 0.99
    V = np.zeros(env.n_states)
    for _ in range(300):
        Q = R[:, None] + gamma * (P @ V)
        V_new = temperature * logsumexp(Q / temperature, axis=1)
        if np.max(np.abs(V_new - V)) < 1e-4:
            break
        V = V_new
    policy = np.exp((Q - V[:, None]) / temperature)
    policy /= policy.sum(axis=1, keepdims=True)
    return policy

# ===========================================================================
# 专家策略
# ===========================================================================
def get_expert_policy(env):
    #n_actions = env.n_actions
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

# ===========================================================================
def evaluate_policy_maxent(env, policy, episodes=200):
    succ = 0
    total_r = 0.0
    total_steps = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        steps = 0
        while not done and steps < 100:
            idx = env._state_to_idx(s)
            a = np.random.choice(env.n_actions, p=policy[idx])
            s, r, done, _ = env.step(a)
            ep_r += r
            steps += 1
        total_r += ep_r
        total_steps += steps
        if env.is_terminal(s):
            succ += 1
    return {
        "success_rate": succ / episodes,
        "mean_reward": total_r / episodes,
        "mean_steps": total_steps / episodes
    }

# ===========================================================================
def evaluate_policy_normal(env, policy, episodes=200):
    success = 0
    total_r = 0.0
    total_steps = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0
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


def evaluate_policy_normal_BT_SUCCESS(env, policy, episodes=200):
    success = 0
    total_r = 0.0
    total_steps = 0
    max_steps = 100  

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        steps = 0

        while not done and steps < max_steps:
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

# ===================== 指标 =====================
def compute_mae(a, b):
    return np.mean(np.abs(a - b))

def compute_corr(a, b):
    return np.corrcoef(a.flatten(), b.flatten())[0, 1]

def compute_sim(pi_true, pi_learned):
    return np.mean(np.argmax(pi_true, axis=1) == np.argmax(pi_learned, axis=1))

# ===================== 专家策略 =====================
def value_iteration(env):
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
    return policy, V

# ===================== 奖励热力图 =====================
def plot_reward_heatmap(reward_mat, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(reward_mat, cmap=CUSTOM_CMAP)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ===================== 单次实验 =====================
def run_single_experiment(env_size=10, seed=42, traj_num=50, pref_num=200, noise_level=0.0):
    np.random.seed(seed)

    env = AdvancedGridWorld({
        "grid_size": env_size,
        "max_steps": 500
        
    })
    n_actions = env.n_actions
    pi_e = get_expert_policy(env)
    r_true = env.true_reward_matrix
    #r_true = env.get_true_reward() 

    trajs = []
    for _ in range(traj_num):
        t = env.generate_expert_dataset(pi_e, 1)[0]
        trajs.append(t)

    prefs = []
    for _ in range(pref_num):
        ti = env.generate_expert_dataset(pi_e, 1)[0]
        tj = env.generate_expert_dataset(pi_e, 1)[0]
        tj["states"] = tj["states"][:len(tj["states"])//2]
        prefs.append((ti, tj, 1.0))
    
    _, r_lp = linear_programming_irl(env, pi_e)
    _, r_mm = maximum_margin_irl(env, pi_e)
    w_me, r_me, _ = maximum_entropy_irl(env, trajs, temperature=0.2, learning_rate=0.15, verbose=True, reg_coeff=1e-5)
    w_bt, r_bt, _ = preference_irl_bt(env, prefs, lr=0.05, n_iterations=1000, verbose=True)
    

    w_hybrid, r_hybrid = maxent_bt_irl(env, trajs, prefs, verbose=True)

    result = {}
    
    result["GT"] = {"reward": r_true}

    # -------------------- LP --------------------
    pi_lp, _ = value_iteration_with_reward(env, r_lp.flatten())
    perf = evaluate_policy_normal(env, pi_lp)
    result["LP"] = {
        "mae": compute_mae(r_true, r_lp), "corr": compute_corr(r_true, r_lp), "sim": compute_sim(pi_e, pi_lp),
        "mean_reward": perf["mean_reward"], "mean_steps": perf["mean_steps"], "success_rate": perf["success_rate"], "reward": r_lp
    }
    # -------------------- MaxMargin --------------------
    pi_mm, _ = value_iteration_with_reward(env, r_mm.flatten())
    perf = evaluate_policy_normal(env, pi_mm)
    result["MaxMargin"] = {
        "mae": compute_mae(r_true, r_mm), "corr": compute_corr(r_true, r_mm), "sim": compute_sim(pi_e, pi_mm),
        "mean_reward": perf["mean_reward"], "mean_steps": perf["mean_steps"], "success_rate": perf["success_rate"], "reward": r_mm
    }
    # -------------------- MaxEnt --------------------
    pi_me = get_policy_from_maxent_weights(env, w_me, temperature=0.75)
    perf = evaluate_policy_maxent(env, pi_me)
    result["MaxEnt"] = {
        "mae": compute_mae(r_true, r_me), "corr": compute_corr(r_true, r_me), "sim": compute_sim(pi_e, pi_me),
        "mean_reward": perf["mean_reward"], "mean_steps": perf["mean_steps"], "success_rate": perf["success_rate"], "reward": r_me
    }
    # -------------------- PbIRL-BT --------------------
    pi_bt, _ = value_iteration_with_reward(env, r_bt.flatten())
    
    perf = evaluate_policy_normal_BT_SUCCESS(env, pi_bt)
    result["BT"] = {
        "mae": compute_mae(r_true, r_bt), "corr": compute_corr(r_true, r_bt), "sim": compute_sim(pi_e, pi_bt),
        "mean_reward": perf["mean_reward"], "mean_steps": perf["mean_steps"], "success_rate": perf["success_rate"], "reward": r_bt
    }
    # -------------------- MaxEnt+BT --------------------
    pi_hybrid, _ = value_iteration_with_reward(env, r_hybrid.flatten())
    perf = evaluate_policy_normal_BT_SUCCESS(env, pi_hybrid)
    result["MaxEntBT"] = {
        "mae": compute_mae(r_true, r_hybrid), "corr": compute_corr(r_true, r_hybrid), "sim": compute_sim(pi_e, pi_hybrid),
        "mean_reward": perf["mean_reward"], "mean_steps": perf["mean_steps"], "success_rate": perf["success_rate"], "reward": r_hybrid
    }

    return result

# ===================== 多次平均 =====================
def average_results(runs):
    keys = ["GT", "LP", "MaxMargin", "MaxEnt", "BT", "MaxEntBT"]
    metrics = ["mae", "corr", "sim", "mean_reward", "mean_steps", "success_rate"]
    avg = {k: {} for k in keys}
    for k in keys:
        if k == "GT":
            avg[k]["reward"] = runs[0][k]["reward"]
        else:
            for m in metrics:
                avg[k][m] = np.mean([r[k][m] for r in runs])
            avg[k]["reward"] = runs[0][k]["reward"]
    return avg

# ===================== 完整消融实验 =====================
def run_all_ablation(num_runs=3):
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    data = {"noise": {}, "traj": {}, "prefs": {}, "grid": {}, "best": None}

    print("Running base...")
    data["best"] = average_results([run_single_experiment(seed=42+i) for i in range(num_runs)])

    print("Running noise...")
    for noise in [0, 0.1, 0.2, 0.3]:
        data["noise"][noise] = average_results([run_single_experiment(seed=42+i, noise_level=noise) for i in range(num_runs)])

    print("Running traj...")
    for n in [20,30,50,100]:
        data["traj"][n] = average_results([run_single_experiment(seed=42+i, traj_num=n) for i in range(num_runs)])

    print("Running prefs...")
    for n in [100,200,300,500]:
        data["prefs"][n] = average_results([run_single_experiment(seed=42+i, pref_num=n) for i in range(num_runs)])

    print("Running grid...")
    for s in [6,8,10]:
        data["grid"][s] = average_results([run_single_experiment(env_size=s, seed=42+i) for i in range(num_runs)])

    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(data, f)
    return data

# ===================== 表格 =====================
def print_table(data):
    print("\n" + "=" * 110)
    print(f"FINAL RESULTS (Averaged over 5 random maps)")
    print("=" * 110)
    res = data["best"]
    rows = [("LP-IRL", res["LP"]), ("MaxMargin-IRL", res["MaxMargin"]), ("MaxEnt-IRL", res["MaxEnt"]), ("PbIRL-BT", res["BT"]), ("MaxEnt+BT", res["MaxEntBT"])]
    print(f"{'Algorithm':<20} | {'MAE':<7} | {'Corr':<7} | {'Sim':<8} | {'Reward':<8} | {'Steps':<7} | {'SR':<8}")
    print("-" * 110)
    for name, d in rows:
        print(f"{name:<20} | {d['mae']:<7.3f} | {d['corr']:<7.3f} | {d['sim']:<7.1%} | {d['mean_reward']:<8.2f} | {d['mean_steps']:<7.1f} | {d['success_rate']:<8.1%}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    
    os.makedirs("result", exist_ok=True)
    print("Running FULL Ablation Experiments...")
    data = run_all_ablation(num_runs=5)
    print_table(data)
    print("\nAll done!")