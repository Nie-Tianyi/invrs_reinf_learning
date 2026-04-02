"""
深度最大熵IRL性能测试（优化后）
"""
import time
import sys
import numpy as np
import torch

from deep_irl_algorithms import deep_maximum_entropy_irl
from environment import AdvancedGridWorld, value_iteration

def main():
    print("=" * 70)
    print("深度最大熵IRL性能测试（优化后）")
    print("=" * 70)
    
    # 1. 初始化环境
    print("\n1. 初始化Advanced GridWorld环境")
    env = AdvancedGridWorld()
    print(f"   网格大小: {env.grid_size}x{env.grid_size}")
    print(f"   总状态数: {env.n_states}")
    
    # 2. 生成专家最优策略
    print("\n2. 生成专家最优策略")
    expert_policy, expert_V = value_iteration(env)
    print(f"   专家策略形状: {expert_policy.shape}")
    
    # 3. 生成专家轨迹数据（减少数量）
    print("\n3. 生成专家轨迹数据")
    expert_trajectories = env.generate_expert_dataset(
        expert_policy=expert_policy,
        n_trajectories=10,  # 减少轨迹数量
        noise_level=0.0,
    )
    print(f"   轨迹数量: {len(expert_trajectories)}")
    avg_len = np.mean([len(traj['states']) for traj in expert_trajectories])
    print(f"   平均轨迹长度: {avg_len:.1f}")
    
    # 4. 运行深度最大熵IRL（小规模测试）
    print("\n4. 运行深度最大熵IRL算法（10次迭代）")
    print("   开始训练...")
    
    start_time = time.time()
    
    network, recovered_reward, losses = deep_maximum_entropy_irl(
        env=env,
        expert_trajectories=expert_trajectories,
        network=None,
        window_size=5,
        gamma=0.99,
        temperature=1.0,
        learning_rate=0.0001,
        n_iterations=10,  # 仅10次迭代
        reg_coeff=0.01,
        verbose=True,  # 打印进度
        device='cpu',  # 使用CPU测试
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n   训练完成!")
    print(f"   网络参数数量: {sum(p.numel() for p in network.parameters())}")
    print(f"   总耗时: {elapsed:.2f}秒")
    print(f"   平均每次迭代: {elapsed/10:.2f}秒")
    print(f"   损失变化: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    # 5. 性能估算
    print("\n5. 性能估算")
    print(f"   预计20次迭代时间: {elapsed/10*20:.2f}秒 (~{elapsed/10*20/60:.1f}分钟)")
    print(f"   预计100次迭代时间: {elapsed/10*100:.2f}秒 (~{elapsed/10*100/60:.1f}分钟)")
    
    # 6. 与原始性能对比
    print("\n6. 性能改进分析")
    print("   原始性能: 20次迭代约3小时 (180分钟)")
    print(f"   优化后性能: 20次迭代预计 {elapsed/10*20/60:.1f}分钟")
    speedup = 180 / (elapsed/10*20/60) if elapsed > 0 else float('inf')
    print(f"   速度提升: {speedup:.1f}倍")
    
    return network, recovered_reward, losses

if __name__ == "__main__":
    main()