"""
深度逆强化学习演示脚本
展示深度特征版本IRL的完整流程，并与线性特征版本进行对比

主要功能：
1. 深度最大熵IRL算法演示
2. 与线性最大熵IRL的对比
3. 深度特征的可视化分析
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple

from environment import AdvancedGridWorld, value_iteration
from irl_algorithms import (
    maximum_entropy_irl,
    compute_expert_feature_expectations_from_trajectories,
    evaluate_reward_recovery,
    visualize_reward_comparison,
)
from deep_irl_algorithms import (
    deep_maximum_entropy_irl,
    evaluate_deep_irl_recovery,
    visualize_deep_irl_results,
    compute_reward_matrix_from_network,
)
from deep_feature_extractor import LocalGridRewardNet, DeepFeatureExtractor


def compare_linear_vs_deep_irl():
    """
    线性特征IRL与深度特征IRL的完整对比实验
    """
    print("=" * 80)
    print("线性特征IRL vs 深度特征IRL 对比实验")
    print("=" * 80)
    
    # 1. 初始化环境（使用固定种子以确保公平比较）
    print("\n1. 初始化环境（固定随机种子）")
    env = AdvancedGridWorld(config={"seed": 42})
    print(f"   网格大小: {env.grid_size}x{env.grid_size}")
    print(f"   总状态数: {env.n_states}")
    print(f"   线性特征维度: {env.feature_matrix.shape[1]}")
    
    # 2. 生成专家最优策略和轨迹
    print("\n2. 生成专家数据")
    expert_policy, expert_V = value_iteration(env)
    
    # 生成专家轨迹（两种算法使用相同数据）
    expert_trajectories = env.generate_expert_dataset(
        expert_policy=expert_policy,
        n_trajectories=100,  # 使用100条轨迹以获得稳定结果
        noise_level=0.0,
    )
    print(f"   轨迹数量: {len(expert_trajectories)}")
    print(f"   平均轨迹长度: {np.mean([len(traj['states']) for traj in expert_trajectories]):.1f}")
    
    # 3. 获取真实奖励矩阵
    true_reward = env.get_true_reward()
    
    # 4. 运行线性最大熵IRL（基线）
    print("\n3. 运行线性最大熵IRL（基线）")
    print("   " + "-" * 50)
    
    weights_linear, reward_linear, losses_linear = maximum_entropy_irl(
        env,
        expert_trajectories,
        gamma=0.99,
        temperature=1.0,
        learning_rate=0.05,
        n_iterations=200,
        reg_coeff=0.01,
        verbose=True,
    )
    
    # 评估线性版本
    metrics_linear = evaluate_reward_recovery(env, reward_linear, true_reward)
    print(f"\n   线性IRL恢复指标:")
    print(f"     MSE（均方误差）: {metrics_linear['mse']:.4f}")
    print(f"     MAE（平均绝对误差）: {metrics_linear['mae']:.4f}")
    print(f"     相关性: {metrics_linear['correlation']:.4f}")
    print(f"     余弦相似度: {metrics_linear['cosine_similarity']:.4f}")
    
    # 5. 运行深度最大熵IRL
    print("\n4. 运行深度最大熵IRL（深度特征）")
    print("   " + "-" * 50)
    
    # 检查是否有GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   使用设备: {device}")
    
    network_deep, reward_deep, losses_deep = deep_maximum_entropy_irl(
        env=env,
        expert_trajectories=expert_trajectories,
        network=None,
        window_size=5,
        gamma=0.99,
        temperature=1.0,
        learning_rate=0.001,
        n_iterations=200,
        reg_coeff=0.01,
        verbose=True,
        device=device,
    )
    
    # 评估深度版本
    metrics_deep = evaluate_deep_irl_recovery(env, reward_deep, true_reward)
    print(f"\n   深度IRL恢复指标:")
    print(f"     MSE（均方误差）: {metrics_deep['mse']:.4f}")
    print(f"     MAE（平均绝对误差）: {metrics_deep['mae']:.4f}")
    print(f"     相关性: {metrics_deep['correlation']:.4f}")
    print(f"     余弦相似度: {metrics_deep['cosine_similarity']:.4f}")
    
    # 6. 对比分析
    print("\n5. 对比分析")
    print("   " + "-" * 50)
    
    # 计算改进百分比（深度相对于线性）
    mse_improvement = (metrics_linear['mse'] - metrics_deep['mse']) / metrics_linear['mse'] * 100
    mae_improvement = (metrics_linear['mae'] - metrics_deep['mae']) / metrics_linear['mae'] * 100
    corr_improvement = (metrics_deep['correlation'] - metrics_linear['correlation']) / abs(metrics_linear['correlation']) * 100
    
    print(f"   MSE改进: {mse_improvement:+.1f}% ({metrics_linear['mse']:.4f} → {metrics_deep['mse']:.4f})")
    print(f"   MAE改进: {mae_improvement:+.1f}% ({metrics_linear['mae']:.4f} → {metrics_deep['mae']:.4f})")
    print(f"   相关性改进: {corr_improvement:+.1f}% ({metrics_linear['correlation']:.4f} → {metrics_deep['correlation']:.4f})")
    
    # 7. 可视化对比结果
    print("\n6. 可视化对比结果")
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 真实奖励
    im1 = axes[0, 0].imshow(true_reward, cmap="RdYlGn", origin="upper")
    axes[0, 0].set_title("Ground Truth Reward", fontsize=14)
    axes[0, 0].set_xlabel("Y")
    axes[0, 0].set_ylabel("X")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 线性IRL恢复奖励
    im2 = axes[0, 1].imshow(reward_linear, cmap="RdYlGn", origin="upper")
    axes[0, 1].set_title(f"Linear IRL Recovery\nMSE={metrics_linear['mse']:.3f}, Corr={metrics_linear['correlation']:.3f}", fontsize=14)
    axes[0, 1].set_xlabel("Y")
    axes[0, 1].set_ylabel("X")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 深度IRL恢复奖励
    im3 = axes[0, 2].imshow(reward_deep, cmap="RdYlGn", origin="upper")
    axes[0, 2].set_title(f"Deep IRL Recovery\nMSE={metrics_deep['mse']:.3f}, Corr={metrics_deep['correlation']:.3f}", fontsize=14)
    axes[0, 2].set_xlabel("Y")
    axes[0, 2].set_ylabel("X")
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. 线性IRL误差
    error_linear = reward_linear - true_reward
    im4 = axes[1, 0].imshow(error_linear, cmap="RdBu", origin="upper", 
                           vmin=-max(abs(error_linear.flatten())), 
                           vmax=max(abs(error_linear.flatten())))
    axes[1, 0].set_title(f"Linear IRL Error\nMAE={metrics_linear['mae']:.3f}", fontsize=14)
    axes[1, 0].set_xlabel("Y")
    axes[1, 0].set_ylabel("X")
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 5. 深度IRL误差
    error_deep = reward_deep - true_reward
    im5 = axes[1, 1].imshow(error_deep, cmap="RdBu", origin="upper",
                           vmin=-max(abs(error_deep.flatten())),
                           vmax=max(abs(error_deep.flatten())))
    axes[1, 1].set_title(f"Deep IRL Error\nMAE={metrics_deep['mae']:.3f}", fontsize=14)
    axes[1, 1].set_xlabel("Y")
    axes[1, 1].set_ylabel("X")
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 6. 训练损失对比
    axes[1, 2].plot(losses_linear, label="Linear IRL", alpha=0.7)
    axes[1, 2].plot(losses_deep, label="Deep IRL", alpha=0.7)
    axes[1, 2].set_title("Training Loss Comparison", fontsize=14)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle("Linear vs Deep IRL: Reward Recovery Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # 8. 深度特征分析
    print("\n7. 深度特征分析")
    
    # 创建深度特征提取器
    deep_extractor = DeepFeatureExtractor(network=network_deep, window_size=5)
    
    # 分析神经网络学习的特征
    print("   分析神经网络学习的奖励模式...")
    
    # 选择几个代表性状态进行分析
    sample_states = [
        (0, 0),  # 起点
        (env.grid_size//2, env.grid_size//2),  # 中心
        (env.goal_state[0], env.goal_state[1]),  # 终点
    ]
    
    for state in sample_states:
        reward_pred = deep_extractor.get_reward(env, state)
        reward_true = true_reward[state]
        print(f"     状态 {state}: 真实奖励={reward_true:.2f}, 预测奖励={reward_pred:.2f}, "
              f"误差={abs(reward_pred - reward_true):.2f}")
    
    # 9. 泛化能力测试（在新地图上）
    print("\n8. 泛化能力测试")
    print("   在新地图配置上测试训练好的深度模型...")
    
    # 创建新环境（不同随机种子）
    env_new = AdvancedGridWorld(config={"seed": 12345})
    true_reward_new = env_new.get_true_reward()
    
    # 使用训练好的网络预测新地图的奖励
    reward_new_pred = compute_reward_matrix_from_network(
        network_deep, env_new, window_size=5, device=device
    )
    
    # 评估泛化性能
    metrics_generalization = evaluate_deep_irl_recovery(
        env_new, reward_new_pred, true_reward_new
    )
    
    print(f"   泛化测试指标:")
    print(f"     MSE: {metrics_generalization['mse']:.4f}")
    print(f"     MAE: {metrics_generalization['mae']:.4f}")
    print(f"     相关性: {metrics_generalization['correlation']:.4f}")
    
    # 10. 保存结果
    print("\n9. 保存结果")
    
    results = {
        'linear_metrics': metrics_linear,
        'deep_metrics': metrics_deep,
        'generalization_metrics': metrics_generalization,
        'improvements': {
            'mse_improvement': mse_improvement,
            'mae_improvement': mae_improvement,
            'corr_improvement': corr_improvement,
        },
        'network_params': sum(p.numel() for p in network_deep.parameters()),
    }
    
    # 保存模型和结果
    torch.save({
        'network_state_dict': network_deep.state_dict(),
        'linear_reward': reward_linear,
        'deep_reward': reward_deep,
        'results': results,
    }, "comparison_results.pth")
    
    print(f"   结果已保存到: comparison_results.pth")
    print(f"   网络参数数量: {results['network_params']}")
    
    print("\n" + "=" * 80)
    print("对比实验完成!")
    print("=" * 80)


def analyze_deep_feature_visualization():
    """
    深度特征可视化分析
    """
    print("\n" + "=" * 80)
    print("深度特征可视化分析")
    print("=" * 80)
    
    # 初始化环境
    env = AdvancedGridWorld(config={"seed": 42})
    
    # 创建随机初始化的深度特征提取器
    deep_extractor = DeepFeatureExtractor(window_size=5)
    
    # 获取预测的奖励矩阵
    reward_matrix = deep_extractor.get_reward_matrix(env)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 真实地形网格
    terrain_colors = ['lightgray', 'black', 'brown', 'lightgreen', 'gold']
    terrain_names = ['Normal', 'Obstacle', 'Mud', 'Grass', 'Goal']
    
    # 创建地形可视化
    terrain_vis = np.zeros((env.grid_size, env.grid_size, 3))
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            terrain = env.grid[i, j]
            if terrain == 0:  # 普通
                terrain_vis[i, j] = [0.9, 0.9, 0.9]
            elif terrain == 1:  # 障碍物
                terrain_vis[i, j] = [0.2, 0.2, 0.2]
            elif terrain == 2:  # 泥潭
                terrain_vis[i, j] = [0.6, 0.4, 0.2]
            elif terrain == 3:  # 草地
                terrain_vis[i, j] = [0.4, 0.8, 0.4]
            else:  # 终点
                terrain_vis[i, j] = [1.0, 0.8, 0.0]
    
    axes[0].imshow(terrain_vis, origin="upper")
    axes[0].set_title("Terrain Map", fontsize=14)
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("X")
    
    # 添加图例
    for i, (color, name) in enumerate(zip(terrain_colors, terrain_names)):
        axes[0].plot([], [], 's', color=color, label=name, markersize=10)
    axes[0].legend(loc='upper right', fontsize=8)
    
    # 2. 神经网络初始预测的奖励
    im2 = axes[1].imshow(reward_matrix, cmap="RdYlGn", origin="upper")
    axes[1].set_title("Neural Network Initial Reward Prediction", fontsize=14)
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("X")
    plt.colorbar(im2, ax=axes[1])
    
    # 3. 局部网格示例
    axes[2].set_title("Local Grid Example", fontsize=14)
    
    # 选择中心状态
    center_state = (env.grid_size//2, env.grid_size//2)
    local_grid = env.get_local_grid(center_state, window_size=5)
    
    # 绘制局部网格
    local_vis = np.zeros((5, 5, 3))
    for i in range(5):
        for j in range(5):
            terrain = local_grid[i, j]
            if terrain == 0:  # 普通
                local_vis[i, j] = [0.9, 0.9, 0.9]
            elif terrain == 1:  # 障碍物
                local_vis[i, j] = [0.2, 0.2, 0.2]
            elif terrain == 2:  # 泥潭
                local_vis[i, j] = [0.6, 0.4, 0.2]
            elif terrain == 3:  # 草地
                local_vis[i, j] = [0.4, 0.8, 0.4]
            else:  # 终点
                local_vis[i, j] = [1.0, 0.8, 0.0]
    
    # 标记中心
    local_vis[2, 2] = [1.0, 0.0, 0.0]  # 红色中心
    
    axes[2].imshow(local_vis, origin="upper")
    axes[2].set_xlabel("Local Y")
    axes[2].set_ylabel("Local X")
    axes[2].grid(True, color='black', linewidth=0.5)
    axes[2].set_xticks(np.arange(-0.5, 5, 1))
    axes[2].set_yticks(np.arange(-0.5, 5, 1))
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])
    
    plt.suptitle("Deep Feature Analysis: Terrain, Initial Predictions, and Local Grid", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()
    
    print("深度特征可视化完成!")


def main():
    """
    主函数：运行深度IRL演示
    """
    print("深度逆强化学习演示脚本")
    print("=" * 60)
    print("选项:")
    print("  1. 线性vs深度IRL对比实验")
    print("  2. 深度特征可视化分析")
    print("  3. 运行完整演示（包含对比和可视化）")
    print()
    
    try:
        choice = int(input("请选择操作 (1-3): "))
    except:
        choice = 3  # 默认运行完整演示
    
    if choice == 1:
        compare_linear_vs_deep_irl()
    elif choice == 2:
        analyze_deep_feature_visualization()
    else:
        # 运行完整演示
        compare_linear_vs_deep_irl()
        analyze_deep_feature_visualization()
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()