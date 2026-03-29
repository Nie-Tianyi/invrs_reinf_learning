from environment import AdvancedGridWorld, value_iteration
import numpy as np

def main():
    # 1. 初始化环境
    print("=== 初始化Advanced GridWorld环境 ===")
    env = AdvancedGridWorld()
    print(f"网格大小：{env.grid_size}x{env.grid_size}")
    print(f"总状态数：{env.n_states}")
    print(f"特征维度：{env.feature_matrix.shape[1]}")

    # 2. 价值迭代生成专家最优策略
    print("\n=== 生成专家最优策略 ===")
    expert_policy, expert_V = value_iteration(env)

    # 3. 生成专家演示数据集（Student1核心交付物1）
    print("\n=== 生成专家演示数据集 ===")
    expert_dataset = env.generate_expert_dataset(
        expert_policy=expert_policy,
        n_trajectories=100,  # 生成100条专家轨迹
        noise_level=0.0,  # 无噪声的完美专家
    )
    print(f"生成专家轨迹数量：{len(expert_dataset)}")
    print(
        f"单条轨迹平均步长：{np.mean([len(traj['states']) for traj in expert_dataset]):.2f}"
    )
    print(
        f"单条轨迹平均总回报：{np.mean([np.sum(traj['rewards']) for traj in expert_dataset]):.2f}"
    )

    # 4. 生成偏好对比数据集（Student1核心交付物2）
    print("\n=== 生成偏好对比数据集 ===")
    preference_dataset = env.generate_preference_dataset(
        expert_policy=expert_policy,
        n_pairs=500,  # 生成500个偏好对
        noise_level=0.0,
    )
    print(f"生成偏好对数量：{len(preference_dataset)}")

    # 5. 可视化：真实奖励热力图
    print("\n=== 绘制真实奖励热力图 ===")
    env.plot_reward_heatmap(title="Ground Truth Reward Heatmap")

    # 6. 可视化：单条专家轨迹
    print("\n=== 绘制专家轨迹示例 ===")
    sample_traj = expert_dataset[0]
    env.plot_trajectory(sample_traj, title="Sample Expert Trajectory")

    # 7. 输出给Student2的核心数据
    print("\n=== 给IRL算法的核心数据 ===")
    print(f"特征矩阵形状：{env.feature_matrix.shape}")
    print(f"转移矩阵形状：{env.transition_matrix.shape}")
    print(f"真实奖励矩阵形状：{env.true_reward_matrix.shape}")


if __name__ == "__main__":
    main()
