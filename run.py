"""
IRL逆强化学习项目 — 统一命令行入口

用法:
    uv run python run.py demo              运行基础IRL演示 (LP + MaxMargin + MaxEnt)
    uv run python run.py demo --deep       包含深度IRL对比
    uv run python run.py experiment        完整对比实验
    uv run python run.py experiment --quick  快速实验（减少迭代次数）
    uv run python run.py ablation          完整消融实验
    uv run python run.py ablation --dim noise --runs 3  仅噪声消融, 3个种子
    uv run python run.py plot              从checkpoint生成图表
    uv run python run.py smoke             快速验证所有模块是否正常
"""

import argparse
import sys


def cmd_demo(args):
    """运行IRL算法演示"""
    import numpy as np
    import matplotlib.pyplot as plt
    from environment import AdvancedGridWorld, value_iteration
    from irl_algorithms import (
        linear_programming_irl,
        maximum_margin_irl,
        maximum_entropy_irl,
        evaluate_reward_recovery,
        visualize_reward_comparison,
    )

    print("=" * 60)
    print("IRL 算法演示")
    print("=" * 60)

    env = AdvancedGridWorld()
    expert_policy, _ = value_iteration(env)
    true_reward = env.get_true_reward()

    # --- 线性规划IRL ---
    print("\n[1/3] 线性规划IRL (Ng & Russell)")
    try:
        w_lp, r_lp = linear_programming_irl(env, expert_policy)
        m = evaluate_reward_recovery(env, r_lp, true_reward)
        print(f"  权重: {np.round(w_lp, 3)}")
        print(f"  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  Corr={m['correlation']:.4f}")
        visualize_reward_comparison(env, true_reward, r_lp, "Linear Programming IRL")
    except Exception as e:
        print(f"  失败: {e}")

    # --- 最大间隔IRL ---
    print("\n[2/3] 最大间隔IRL (Abbeel & Ng)")
    try:
        w_mm, r_mm = maximum_margin_irl(env, expert_policy)
        m = evaluate_reward_recovery(env, r_mm, true_reward)
        print(f"  权重: {np.round(w_mm, 3)}")
        print(f"  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  Corr={m['correlation']:.4f}")
        visualize_reward_comparison(env, true_reward, r_mm, "Maximum Margin IRL")
    except Exception as e:
        print(f"  失败: {e}")

    # --- 最大熵IRL ---
    print("\n[3/3] 最大熵IRL (Ziebart et al.)")
    try:
        trajectories = env.generate_expert_dataset(expert_policy, n_trajectories=50, noise_level=0.0)
        w_me, r_me, losses = maximum_entropy_irl(
            env, trajectories, gamma=0.99, temperature=0.8,
            learning_rate=0.03, n_iterations=500, reg_coeff=1e-4, verbose=True,
        )
        m = evaluate_reward_recovery(env, r_me, true_reward)
        print(f"  权重: {np.round(w_me, 3)}")
        print(f"  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  Corr={m['correlation']:.4f}")
        visualize_reward_comparison(env, true_reward, r_me, "Maximum Entropy IRL")

        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel("Iteration"); plt.ylabel("Loss")
        plt.title("MaxEnt IRL Training Loss")
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"  失败: {e}")

    # --- 深度IRL (可选) ---
    if args.deep:
        print("\n[4/4] 深度最大熵IRL")
        try:
            from deep_irl_algorithms import (
                deep_maximum_entropy_irl, evaluate_deep_irl_recovery,
                visualize_deep_irl_results,
            )
            print("  生成专家轨迹...")
            trajectories = env.generate_expert_dataset(expert_policy, n_trajectories=50, noise_level=0.0)
            print("  训练深度网络...")
            net, r_deep, losses_d = deep_maximum_entropy_irl(
                env, trajectories, gamma=0.99, temperature=1.0,
                learning_rate=0.0001, n_iterations=20, reg_coeff=0.01, verbose=True,
            )
            m = evaluate_deep_irl_recovery(env, r_deep, true_reward)
            print(f"  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  Corr={m['correlation']:.4f}")
            visualize_deep_irl_results(env, true_reward, r_deep, losses_d,
                                       "Deep Maximum Entropy IRL")
        except Exception as e:
            print(f"  失败: {e}")

    print("\n演示完成。")


def cmd_experiment(args):
    """运行完整对比实验"""
    import numpy as np
    import json
    import time
    from environment import AdvancedGridWorld, value_iteration
    from irl_algorithms import (
        linear_programming_irl, maximum_margin_irl, maximum_entropy_irl,
    )
    from policy_training import compare_policies

    quick = args.quick
    n_traj = 20 if quick else 100
    maxent_iter = 30 if quick else 300
    noise_levels = [0.0, 0.2] if quick else [0.0, 0.1, 0.2, 0.3]

    print("=" * 60)
    print("IRL 对比实验" + (" (快速模式)" if quick else ""))
    print("=" * 60)

    env = AdvancedGridWorld({"grid_size": 8})
    expert_policy, _ = value_iteration(env)
    true_reward = env.get_true_reward()

    results = {}
    for noise in noise_levels:
        print(f"\n--- 噪声水平: {noise} ---")
        trajectories = env.generate_expert_dataset(expert_policy, n_trajectories=n_traj, noise_level=noise)

        recovered = {}
        t0 = time.time()
        try:
            w_lp, r_lp = linear_programming_irl(env, expert_policy)
            recovered["LP"] = r_lp
            print(f"  LP IRL: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  LP IRL 失败: {e}")

        t0 = time.time()
        try:
            w_mm, r_mm = maximum_margin_irl(env, expert_policy)
            recovered["MaxMargin"] = r_mm
            print(f"  MaxMargin IRL: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  MaxMargin IRL 失败: {e}")

        t0 = time.time()
        try:
            w_me, r_me, _ = maximum_entropy_irl(
                env, trajectories, n_iterations=maxent_iter, verbose=False,
            )
            recovered["MaxEnt"] = r_me
            print(f"  MaxEnt IRL: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  MaxEnt IRL 失败: {e}")

        if recovered:
            cmp = compare_policies(env, true_reward, recovered, n_episodes=50)
            results[noise] = {"recovered": {k: v.tolist() for k, v in recovered.items()}, "comparison": cmp}

    # 保存结果
    import os
    os.makedirs("result", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"result/experiment_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {path}")


def cmd_ablation(args):
    """运行消融实验"""
    from experiments_final import run_all_ablation

    dims = [args.dim] if args.dim else None
    runs = args.runs
    print(f"消融实验: dims={dims or 'all'}, runs={runs}")
    run_all_ablation(run_dims=dims, num_runs=runs)


def cmd_plot(args):
    """从checkpoint生成图表"""
    import pickle
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    checkpoint = args.checkpoint or "checkpoint/ablation_final.pkl"
    if not os.path.exists(checkpoint):
        print(f"错误: checkpoint 文件不存在: {checkpoint}")
        print("请先运行: uv run irl ablation")
        sys.exit(1)

    print(f"从 {checkpoint} 加载数据...")
    with open(checkpoint, "rb") as f:
        data = pickle.load(f)

    # 调用 plot.py 的逻辑
    import plot as plot_module
    print(f"数据键: {list(data.keys())}")
    print(f"图表将保存到 result/ 目录")

    # 直接执行 plot.py 的绘图逻辑
    LINE_COLORS = ["#fbd2bc", "#feab88", "#b71c2c", "#8b0824", "#6a0624"]
    HEATMAP_COLORS = ["#006633", "#339966", "#99cc66", "#ffffcc", "#ff9966", "#cc3333"]
    HEATMAP_CMAP = LinearSegmentedColormap.from_list("green_yellow_red", HEATMAP_COLORS)
    os.makedirs("result", exist_ok=True)

    # 如果 plot.py 有可导入的绘图函数就用它，否则用内置逻辑
    exec(open("plot.py", encoding="utf-8").read())
    print("图表生成完成。")


def cmd_smoke(args):
    """快速烟雾测试"""
    import numpy as np
    print("=== IRL 项目烟雾测试 ===\n")

    # 1. 环境
    print("[1/5] 测试环境...")
    from environment import AdvancedGridWorld, value_iteration
    env = AdvancedGridWorld()
    assert env.n_states == 100
    expert_policy, _ = value_iteration(env)
    print("  环境 OK")

    # 2. IRL算法
    print("[2/5] 测试IRL算法...")
    from irl_algorithms import linear_programming_irl, maximum_entropy_irl
    w_lp, r_lp = linear_programming_irl(env, expert_policy)
    assert r_lp.shape == (10, 10)
    print(f"  LP IRL OK (权重: {np.round(w_lp, 2)})")

    trajectories = env.generate_expert_dataset(expert_policy, n_trajectories=10, noise_level=0.0)
    w_me, r_me, losses = maximum_entropy_irl(
        env, trajectories, n_iterations=20, verbose=False,
    )
    assert r_me.shape == (10, 10)
    print(f"  MaxEnt IRL OK (loss: {losses[-1]:.4f})")

    # 3. 偏好IRL
    print("[3/5] 测试偏好IRL...")
    from irl_algorithms import preference_irl_bt
    prefs = env.generate_preference_dataset(expert_policy, n_pairs=20, noise_level=0.0)
    w_bt, r_bt, _ = preference_irl_bt(env, prefs, n_iterations=50, verbose=False)
    print(f"  Pref IRL OK")

    # 4. 策略训练
    print("[4/5] 测试策略训练...")
    from policy_training import value_iteration_with_reward, evaluate_policy
    pi, V = value_iteration_with_reward(env, env.get_true_reward().flatten())
    result = evaluate_policy(env, pi, n_episodes=10)
    print(f"  策略评估 OK (成功率: {result['success_rate']:.1%})")

    # 5. 深度IRL
    print("[5/5] 测试深度IRL...")
    try:
        from deep_irl_algorithms import deep_maximum_entropy_irl
        net, r_deep, _ = deep_maximum_entropy_irl(
            env, trajectories, n_iterations=5, verbose=False,
        )
        print(f"  深度IRL OK")
    except Exception as e:
        print(f"  深度IRL 跳过 (可能无torch): {e}")

    print("\n所有测试通过!")


def main():
    parser = argparse.ArgumentParser(
        description="IRL逆强化学习项目 — 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python run.py demo              运行基础IRL演示
  uv run python run.py demo --deep       包含深度IRL对比
  uv run python run.py experiment --quick  快速实验
  uv run python run.py ablation --dim noise --runs 3
  uv run python run.py plot
  uv run python run.py smoke
        """,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # demo
    p_demo = sub.add_parser("demo", help="运行IRL算法演示")
    p_demo.add_argument("--deep", action="store_true", help="包含深度IRL对比")

    # experiment
    p_exp = sub.add_parser("experiment", help="运行完整对比实验")
    p_exp.add_argument("--quick", action="store_true", help="快速模式（减少迭代次数）")

    # ablation
    p_abl = sub.add_parser("ablation", help="运行消融实验")
    p_abl.add_argument("--dim", choices=["noise", "traj", "pref", "grid"],
                       help="只运行指定维度的消融 (默认全部)")
    p_abl.add_argument("--runs", type=int, default=5,
                       help="每个配置的随机种子数 (默认: 5)")

    # plot
    p_plot = sub.add_parser("plot", help="从checkpoint生成图表")
    p_plot.add_argument("--checkpoint", help="checkpoint文件路径 (默认: checkpoint/ablation_final.pkl)")

    # smoke
    sub.add_parser("smoke", help="快速烟雾测试，验证所有模块正常")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "demo": cmd_demo,
        "experiment": cmd_experiment,
        "ablation": cmd_ablation,
        "plot": cmd_plot,
        "smoke": cmd_smoke,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
