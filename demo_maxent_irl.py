"""
最大熵逆强化学习演示脚本（Student3 任务）
展示最大熵IRL算法的完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import AdvancedGridWorld, value_iteration
from irl_algorithms import (
    maximum_entropy_irl,
    compute_expert_feature_expectations_from_trajectories,
    evaluate_reward_recovery,
    visualize_reward_comparison,
)


def main():
    print("=" * 70)
    print("最大熵逆强化学习演示 (Maximum Entropy IRL)")
    print("=" * 70)
    
    # 1. 初始化环境
    print("\n1. 初始化Advanced GridWorld环境")
    env = AdvancedGridWorld()
    print(f"   网格大小: {env.grid_size}x{env.grid_size}")
    print(f"   总状态数: {env.n_states}")
    print(f"   特征维度: {env.feature_matrix.shape[1]}")
    
    # 2. 生成专家最优策略
    print("\n2. 生成专家最优策略")
    expert_policy, expert_V = value_iteration(env)
    print(f"   专家策略形状: {expert_policy.shape}")
    
    # 3. 生成专家轨迹数据
    print("\n3. 生成专家轨迹数据")
    expert_trajectories = env.generate_expert_dataset(
        expert_policy=expert_policy,
        n_trajectories=100,  # 使用100条轨迹以获得稳定特征期望
        noise_level=0.0,
    )
    print(f"   轨迹数量: {len(expert_trajectories)}")
    print(f"   平均轨迹长度: {np.mean([len(traj['states']) for traj in expert_trajectories]):.1f}")
    print(f"   平均轨迹回报: {np.mean([np.sum(traj['rewards']) for traj in expert_trajectories]):.1f}")
    
    # 4. 计算专家特征期望
    print("\n4. 计算专家特征期望")
    mu_expert = compute_expert_feature_expectations_from_trajectories(
        env, expert_trajectories, gamma=0.99
    )
    print(f"   专家特征期望: {mu_expert}")
    
    # 5. 可视化真实奖励地图
    print("\n5. 可视化真实奖励地图")
    true_reward = env.get_true_reward()
    env.plot_reward_heatmap(true_reward, title="Ground Truth Reward Map")
    
    # 6. 运行最大熵IRL算法
    print("\n6. 运行最大熵IRL算法")
    print("   开始训练...")
    
    weights_me, reward_me, losses = maximum_entropy_irl(
        env,
        expert_trajectories,
        gamma=0.99,
        temperature=1.0,
        learning_rate=0.05,  # 较小的学习率以稳定训练
        n_iterations=200,    # 更多迭代以获得更好收敛
        reg_coeff=0.01,
        verbose=True,
    )
    
    print(f"\n   训练完成!")
    print(f"   最终权重: {weights_me}")
    print(f"   权重范数: {np.linalg.norm(weights_me):.4f}")
    
    # 7. 评估恢复效果
    print("\n7. 评估恢复效果")
    metrics = evaluate_reward_recovery(env, reward_me, true_reward)
    print(f"   均方误差 (MSE): {metrics['mse']:.4f}")
    print(f"   平均绝对误差 (MAE): {metrics['mae']:.4f}")
    print(f"   皮尔逊相关性: {metrics['correlation']:.4f}")
    print(f"   余弦相似度: {metrics['cosine_similarity']:.4f}")
    
    # 8. 可视化对比
    print("\n8. 可视化对比真实奖励与恢复奖励")
    visualize_reward_comparison(
        env, true_reward, reward_me, 
        title="Maximum Entropy IRL: Ground Truth vs Recovered Reward"
    )
    
    # 9. 绘制训练过程
    print("\n9. 绘制训练过程")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel("迭代次数")
    axes[0, 0].set_ylabel("损失")
    axes[0, 0].set_title("训练损失曲线")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 损失对数曲线
    axes[0, 1].semilogy(losses)
    axes[0, 1].set_xlabel("迭代次数")
    axes[0, 1].set_ylabel("损失 (对数尺度)")
    axes[0, 1].set_title("训练损失对数曲线")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 权重变化
    axes[1, 0].bar(range(len(weights_me)), weights_me)
    axes[1, 0].set_xlabel("特征索引")
    axes[1, 0].set_ylabel("权重值")
    axes[1, 0].set_title("恢复的奖励权重")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 特征期望对比
    # 计算当前策略的特征期望
    from irl_algorithms import soft_value_iteration, compute_soft_policy, compute_policy_feature_expectations_maxent
    Q, V = soft_value_iteration(env, weights_me, gamma=0.99, temperature=1.0)
    policy = compute_soft_policy(Q, temperature=1.0)
    mu_policy = compute_policy_feature_expectations_maxent(env, policy, gamma=0.99)
    
    x = np.arange(len(mu_expert))
    width = 0.35
    axes[1, 1].bar(x - width/2, mu_expert, width, label='专家特征期望', alpha=0.8)
    axes[1, 1].bar(x + width/2, mu_policy, width, label='策略特征期望', alpha=0.8)
    axes[1, 1].set_xlabel("特征索引")
    axes[1, 1].set_ylabel("特征期望值")
    axes[1, 1].set_title("特征期望对比")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("最大熵IRL训练分析", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 10. 参数敏感性分析（可选）
    print("\n10. 参数敏感性分析（温度参数）")
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    for temp in temperatures:
        print(f"   测试温度参数: {temp}")
        try:
            weights_temp, reward_temp, losses_temp = maximum_entropy_irl(
                env,
                expert_trajectories[:20],  # 使用少量轨迹以加快测试
                gamma=0.99,
                temperature=temp,
                learning_rate=0.05,
                n_iterations=50,
                reg_coeff=0.01,
                verbose=False,
            )
            
            metrics_temp = evaluate_reward_recovery(env, reward_temp, true_reward)
            results.append({
                'temperature': temp,
                'mse': metrics_temp['mse'],
                'correlation': metrics_temp['correlation'],
                'weights_norm': np.linalg.norm(weights_temp),
            })
            print(f"     MSE: {metrics_temp['mse']:.4f}, 相关性: {metrics_temp['correlation']:.4f}")
        except Exception as e:
            print(f"     温度 {temp} 测试失败: {e}")
    
    if results:
        print("\n   温度参数敏感性总结:")
        for r in results:
            print(f"     温度={r['temperature']}: MSE={r['mse']:.4f}, "
                  f"相关性={r['correlation']:.4f}, 权重范数={r['weights_norm']:.4f}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()