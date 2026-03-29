"""
逆强化学习算法演示脚本（Student2 交付物）
展示线性规划IRL和最大间隔IRL的完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import AdvancedGridWorld, value_iteration
from irl_algorithms import (
    linear_programming_irl,
    linear_programming_irl_ng_russell,
    maximum_margin_irl,
    evaluate_reward_recovery,
    visualize_reward_comparison,
)


def main():
    print("=" * 60)
    print("逆强化学习算法演示")
    print("=" * 60)
    
    # 1. 初始化环境
    print("\n1. 初始化Advanced GridWorld环境")
    env = AdvancedGridWorld()
    print(f"   网格大小: {env.grid_size}x{env.grid_size}")
    print(f"   总状态数: {env.n_states}")
    print(f"   特征维度: {env.feature_matrix.shape[1]}")
    
    # 2. 生成专家最优策略（使用价值迭代）
    print("\n2. 生成专家最优策略")
    expert_policy, expert_V = value_iteration(env)
    print(f"   专家策略形状: {expert_policy.shape}")
    print(f"   专家价值函数范围: [{expert_V.min():.2f}, {expert_V.max():.2f}]")
    
    # 3. 可视化真实奖励地图
    print("\n3. 可视化真实奖励地图")
    true_reward = env.get_true_reward()
    env.plot_reward_heatmap(true_reward, title="Ground Truth Reward Map")
    
    # 4. 线性规划IRL（基础版本）
    print("\n4. 线性规划IRL（基础版本）")
    try:
        weights_lp, reward_lp = linear_programming_irl(env, expert_policy, gamma=0.99, margin=1.0)
        print(f"   恢复的权重: {weights_lp}")
        
        metrics = evaluate_reward_recovery(env, reward_lp, true_reward)
        print(f"   恢复指标:")
        print(f"     MSE（均方误差）: {metrics['mse']:.4f}")
        print(f"     MAE（平均绝对误差）: {metrics['mae']:.4f}")
        print(f"     相关性: {metrics['correlation']:.4f}")
        print(f"     余弦相似度: {metrics['cosine_similarity']:.4f}")
        
        # 可视化对比
        visualize_reward_comparison(
            env, true_reward, reward_lp, 
            title="Linear Programming IRL (Basic) Recovery Comparison"
        )
    except Exception as e:
        print(f"   算法失败: {e}")
    
    # 5. 线性规划IRL（Ng & Russell 精确版本）
    print("\n5. 线性规划IRL（Ng & Russell 精确版本）")
    try:
        weights_lp2, reward_lp2 = linear_programming_irl_ng_russell(
            env, expert_policy, gamma=0.99, margin=0.1, reward_bound=50.0
        )
        print(f"   恢复的权重: {weights_lp2}")
        
        metrics = evaluate_reward_recovery(env, reward_lp2, true_reward)
        print(f"   恢复指标:")
        print(f"     MSE（均方误差）: {metrics['mse']:.4f}")
        print(f"     MAE（平均绝对误差）: {metrics['mae']:.4f}")
        print(f"     相关性: {metrics['correlation']:.4f}")
        print(f"     余弦相似度: {metrics['cosine_similarity']:.4f}")
        
        # 可视化对比
        visualize_reward_comparison(
            env, true_reward, reward_lp2, 
            title="Linear Programming IRL (Ng & Russell) Recovery Comparison"
        )
    except Exception as e:
        print(f"   算法失败: {e}")
    
    # 6. 最大间隔IRL
    print("\n6. 最大间隔IRL")
    try:
        weights_mm, reward_mm = maximum_margin_irl(env, expert_policy, gamma=0.99, margin=1.0)
        print(f"   恢复的权重: {weights_mm}")
        
        metrics = evaluate_reward_recovery(env, reward_mm, true_reward)
        print(f"   恢复指标:")
        print(f"     MSE（均方误差）: {metrics['mse']:.4f}")
        print(f"     MAE（平均绝对误差）: {metrics['mae']:.4f}")
        print(f"     相关性: {metrics['correlation']:.4f}")
        print(f"     余弦相似度: {metrics['cosine_similarity']:.4f}")
        
        # 可视化对比
        visualize_reward_comparison(
            env, true_reward, reward_mm, 
            title="Maximum Margin IRL Recovery Comparison"
        )
    except Exception as e:
        print(f"   算法失败: {e}")
    
    # 7. 算法比较总结
    print("\n" + "=" * 60)
    print("算法性能总结")
    print("=" * 60)
    
    # 收集所有恢复的奖励矩阵
    recovered_rewards = []
    algorithm_names = []
    
    if 'reward_lp' in locals():
        recovered_rewards.append(reward_lp)
        algorithm_names.append("Linear Programming IRL (Basic)")
    
    if 'reward_lp2' in locals():
        recovered_rewards.append(reward_lp2)
        algorithm_names.append("Linear Programming IRL (Ng & Russell)")
    
    if 'reward_mm' in locals():
        recovered_rewards.append(reward_mm)
        algorithm_names.append("Maximum Margin IRL")
    
    if recovered_rewards:
        print("\n算法恢复效果对比:")
        for i, (name, rec_reward) in enumerate(zip(algorithm_names, recovered_rewards)):
            metrics = evaluate_reward_recovery(env, rec_reward, true_reward)
            print(f"\n{name}:")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  相关性: {metrics['correlation']:.4f}")
    
    # 8. 可视化所有恢复的奖励（子图）
    if len(recovered_rewards) > 0:
        print("\n生成综合对比图...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 真实奖励
        im0 = axes[0].imshow(true_reward, cmap="RdYlGn", origin="upper")
        axes[0].set_title("Ground Truth Reward", fontsize=12)
        axes[0].set_xlabel("Y Coordinate", fontsize=10)
        axes[0].set_ylabel("X Coordinate", fontsize=10)
        plt.colorbar(im0, ax=axes[0], label="Reward Value")
        
        # 恢复的奖励
        for i, (name, rec_reward) in enumerate(zip(algorithm_names, recovered_rewards)):
            im = axes[i + 1].imshow(rec_reward, cmap="RdYlGn", origin="upper")
            axes[i + 1].set_title(f"{name}", fontsize=12)
            axes[i + 1].set_xlabel("Y Coordinate", fontsize=10)
            axes[i + 1].set_ylabel("X Coordinate", fontsize=10)
            plt.colorbar(im, ax=axes[i + 1], label="Reward Value")
        
        # 隐藏多余的子图
        for j in range(len(recovered_rewards) + 1, 4):
            axes[j].axis("off")
        
        plt.suptitle("IRL算法恢复奖励对比", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()