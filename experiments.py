"""
实验脚本：运行完整的IRL算法比较和策略性能评估
（Student4 任务 - 完整比较实验）
"""

import numpy as np
import json
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from environment import AdvancedGridWorld, value_iteration
from irl_algorithms import (
    linear_programming_irl,
    maximum_margin_irl,
    maximum_entropy_irl,
)
from policy_training import (
    compare_policies,
    compute_policy_similarity,
    evaluate_policy,
    value_iteration_with_reward,
)


def run_experiment(
    env_config: Dict = None,
    n_trajectories: int = 100,
    noise_levels: List[float] = None,
    gamma: float = 0.99,
    n_episodes_eval: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    运行完整实验：在不同噪声水平下测试IRL算法恢复效果和策略性能
    
    :param env_config: 环境配置字典
    :param n_trajectories: 专家轨迹数量（用于最大熵IRL）
    :param noise_levels: 要测试的噪声水平列表
    :param gamma: 折扣因子
    :param n_episodes_eval: 每个策略评估的回合数
    :param seed: 随机种子
    :return: 包含所有实验结果的字典
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    if env_config is None:
        env_config = {"grid_size": 8, "seed": seed}  # 较小网格以加快实验速度
    
    np.random.seed(seed)
    results = {
        "config": {
            "env_config": env_config,
            "n_trajectories": n_trajectories,
            "noise_levels": noise_levels,
            "gamma": gamma,
            "n_episodes_eval": n_episodes_eval,
            "seed": seed,
        },
        "experiments": {},
    }
    
    # 对每个噪声水平运行实验
    for noise in noise_levels:
        print(f"\n{'='*70}")
        print(f"运行实验：噪声水平 = {noise}")
        print(f"{'='*70}")
        
        # 初始化环境
        env = AdvancedGridWorld(env_config)
        true_reward = env.get_true_reward()
        
        # 生成专家策略（使用真实奖励）
        expert_policy, expert_V = value_iteration(env)
        
        # 生成专家轨迹（用于最大熵IRL）
        expert_trajectories = env.generate_expert_dataset(
            expert_policy=expert_policy,
            n_trajectories=n_trajectories,
            noise_level=noise,
        )
        
        # 运行不同IRL算法
        recovered_rewards = {}
        algorithms = [
            ("Linear Programming IRL", linear_programming_irl),
            ("Maximum Margin IRL", maximum_margin_irl),
            ("Maximum Entropy IRL", lambda e, p: maximum_entropy_irl(e, expert_trajectories, verbose=False)),
        ]
        
        for algo_name, algo_func in algorithms:
            print(f"\n  --- {algo_name} ---")
            try:
                start_time = time.time()
                
                if algo_name == "Maximum Entropy IRL":
                    weights, reward, losses = algo_func(env, expert_policy)
                else:
                    weights, reward = algo_func(env, expert_policy)
                
                elapsed = time.time() - start_time
                
                # 评估奖励恢复质量
                reward_diff = np.mean(np.abs(reward - true_reward))
                reward_corr = np.corrcoef(reward.flatten(), true_reward.flatten())[0, 1]
                
                recovered_rewards[algo_name] = reward
                
                print(f"    恢复时间: {elapsed:.2f}秒")
                print(f"    平均绝对误差: {reward_diff:.4f}")
                print(f"    相关性: {reward_corr:.4f}")
                
                # 存储结果
                if noise not in results["experiments"]:
                    results["experiments"][noise] = {}
                
                results["experiments"][noise][algo_name] = {
                    "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
                    "reward_diff": float(reward_diff),
                    "reward_corr": float(reward_corr),
                    "time": elapsed,
                    "success": True,
                }
                
            except Exception as e:
                print(f"    算法失败: {e}")
                if noise not in results["experiments"]:
                    results["experiments"][noise] = {}
                results["experiments"][noise][algo_name] = {
                    "success": False,
                    "error": str(e),
                }
        
        # 比较策略性能（如果至少有一个算法成功）
        if recovered_rewards:
            print(f"\n  --- 策略性能比较 ---")
            try:
                policy_results = compare_policies(
                    env, true_reward, recovered_rewards,
                    gamma=gamma, n_episodes=n_episodes_eval
                )
                
                # 存储策略结果
                for algo_name, data in policy_results.items():
                    if algo_name == "ground_truth":
                        continue
                    if algo_name in results["experiments"][noise]:
                        results["experiments"][noise][algo_name]["policy_metrics"] = data.get("metrics")
                        
                        # 计算策略相似度
                        if data.get("policy") is not None and policy_results["ground_truth"]["policy"] is not None:
                            similarity = compute_policy_similarity(
                                policy_results["ground_truth"]["policy"],
                                data["policy"]
                            )
                            results["experiments"][noise][algo_name]["policy_similarity"] = float(similarity)
                
                # 添加真实策略指标
                if "ground_truth" not in results["experiments"][noise]:
                    results["experiments"][noise]["ground_truth"] = {}
                results["experiments"][noise]["ground_truth"]["policy_metrics"] = \
                    policy_results["ground_truth"]["metrics"]
                    
            except Exception as e:
                print(f"    策略比较失败: {e}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析实验结果，生成汇总统计和洞察
    
    :param results: 实验结果的字典
    :return: 分析结果的字典
    """
    analysis = {
        "summary": {},
        "best_algorithms": {},
        "noise_impact": {},
        "recommendations": [],
    }
    
    config = results["config"]
    noise_levels = config["noise_levels"]
    
    # 汇总每个算法在不同噪声水平下的表现
    algorithms = ["Linear Programming IRL", "Maximum Margin IRL", "Maximum Entropy IRL"]
    
    for algo in algorithms:
        algo_data = []
        for noise in noise_levels:
            if noise in results["experiments"] and algo in results["experiments"][noise]:
                data = results["experiments"][noise][algo]
                if data.get("success", False):
                    algo_data.append({
                        "noise": noise,
                        "reward_diff": data.get("reward_diff", float("inf")),
                        "reward_corr": data.get("reward_corr", -1),
                        "time": data.get("time", float("inf")),
                        "policy_similarity": data.get("policy_similarity", 0),
                    })
        
        if algo_data:
            analysis["summary"][algo] = {
                "avg_reward_diff": np.mean([d["reward_diff"] for d in algo_data]),
                "avg_reward_corr": np.mean([d["reward_corr"] for d in algo_data]),
                "avg_time": np.mean([d["time"] for d in algo_data]),
                "avg_policy_similarity": np.mean([d["policy_similarity"] for d in algo_data]),
                "robustness": len(algo_data) / len(noise_levels),  # 算法成功运行的比例
            }
    
    # 找出最佳算法（根据不同的指标）
    if analysis["summary"]:
        # 基于奖励恢复质量
        best_reward = min(analysis["summary"].items(), 
                         key=lambda x: x[1]["avg_reward_diff"])
        analysis["best_algorithms"]["reward_recovery"] = {
            "algorithm": best_reward[0],
            "avg_reward_diff": best_reward[1]["avg_reward_diff"],
        }
        
        # 基于策略相似度
        best_similarity = max(analysis["summary"].items(),
                             key=lambda x: x[1]["avg_policy_similarity"])
        analysis["best_algorithms"]["policy_similarity"] = {
            "algorithm": best_similarity[0],
            "avg_similarity": best_similarity[1]["avg_policy_similarity"],
        }
        
        # 基于计算效率
        best_time = min(analysis["summary"].items(),
                       key=lambda x: x[1]["avg_time"])
        analysis["best_algorithms"]["efficiency"] = {
            "algorithm": best_time[0],
            "avg_time": best_time[1]["avg_time"],
        }
        
        # 基于鲁棒性（对噪声的抵抗力）
        best_robustness = max(analysis["summary"].items(),
                             key=lambda x: x[1]["robustness"])
        analysis["best_algorithms"]["robustness"] = {
            "algorithm": best_robustness[0],
            "robustness": best_robustness[1]["robustness"],
        }
    
    # 分析噪声影响
    for noise in noise_levels:
        if noise in results["experiments"]:
            noise_data = {}
            for algo in algorithms:
                if algo in results["experiments"][noise]:
                    data = results["experiments"][noise][algo]
                    if data.get("success", False):
                        noise_data[algo] = {
                            "reward_diff": data.get("reward_diff"),
                            "policy_similarity": data.get("policy_similarity"),
                        }
            analysis["noise_impact"][noise] = noise_data
    
    # 生成建议
    if analysis["best_algorithms"]:
        analysis["recommendations"].append(
            f"对于高质量奖励恢复，推荐使用 {analysis['best_algorithms']['reward_recovery']['algorithm']} "
            f"(平均绝对误差: {analysis['best_algorithms']['reward_recovery']['avg_reward_diff']:.4f})"
        )
        analysis["recommendations"].append(
            f"对于策略性能相似度，推荐使用 {analysis['best_algorithms']['policy_similarity']['algorithm']} "
            f"(平均相似度: {analysis['best_algorithms']['policy_similarity']['avg_similarity']:.2%})"
        )
        analysis["recommendations"].append(
            f"对于计算效率，推荐使用 {analysis['best_algorithms']['efficiency']['algorithm']} "
            f"(平均时间: {analysis['best_algorithms']['efficiency']['avg_time']:.2f}秒)"
        )
        analysis["recommendations"].append(
            f"对于噪声环境鲁棒性，推荐使用 {analysis['best_algorithms']['robustness']['algorithm']} "
            f"(成功比例: {analysis['best_algorithms']['robustness']['robustness']:.2%})"
        )
    
    return analysis


def visualize_results(results: Dict[str, Any], analysis: Dict[str, Any], save_dir: str = "."):
    """
    可视化实验结果
    
    :param results: 实验结果
    :param analysis: 分析结果
    :param save_dir: 保存图像的目录
    """
    config = results["config"]
    noise_levels = config["noise_levels"]
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. 奖励恢复误差 vs 噪声水平
    ax = axes[0]
    algorithms = ["Linear Programming IRL", "Maximum Margin IRL", "Maximum Entropy IRL"]
    colors = ['blue', 'green', 'red']
    
    for algo, color in zip(algorithms, colors):
        reward_diffs = []
        for noise in noise_levels:
            if (noise in results["experiments"] and 
                algo in results["experiments"][noise] and
                results["experiments"][noise][algo].get("success", False)):
                reward_diffs.append(results["experiments"][noise][algo]["reward_diff"])
            else:
                reward_diffs.append(np.nan)
        
        ax.plot(noise_levels, reward_diffs, 'o-', color=color, label=algo, linewidth=2)
    
    ax.set_xlabel("噪声水平", fontsize=12)
    ax.set_ylabel("平均绝对误差", fontsize=12)
    ax.set_title("奖励恢复误差 vs 噪声水平", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 策略相似度 vs 噪声水平
    ax = axes[1]
    for algo, color in zip(algorithms, colors):
        similarities = []
        for noise in noise_levels:
            if (noise in results["experiments"] and 
                algo in results["experiments"][noise] and
                results["experiments"][noise][algo].get("success", False)):
                sim = results["experiments"][noise][algo].get("policy_similarity", 0)
                similarities.append(sim)
            else:
                similarities.append(np.nan)
        
        ax.plot(noise_levels, similarities, 'o-', color=color, label=algo, linewidth=2)
    
    ax.set_xlabel("噪声水平", fontsize=12)
    ax.set_ylabel("策略相似度", fontsize=12)
    ax.set_title("策略相似度 vs 噪声水平", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. 计算时间 vs 噪声水平
    ax = axes[2]
    for algo, color in zip(algorithms, colors):
        times = []
        for noise in noise_levels:
            if (noise in results["experiments"] and 
                algo in results["experiments"][noise] and
                results["experiments"][noise][algo].get("success", False)):
                times.append(results["experiments"][noise][algo]["time"])
            else:
                times.append(np.nan)
        
        ax.plot(noise_levels, times, 'o-', color=color, label=algo, linewidth=2)
    
    ax.set_xlabel("噪声水平", fontsize=12)
    ax.set_ylabel("计算时间 (秒)", fontsize=12)
    ax.set_title("计算时间 vs 噪声水平", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. 算法鲁棒性（成功比例）
    ax = axes[3]
    if analysis["summary"]:
        algorithms = list(analysis["summary"].keys())
        robustness = [analysis["summary"][algo]["robustness"] for algo in algorithms]
        
        bars = ax.bar(algorithms, robustness, color=['blue', 'green', 'red'])
        ax.set_xlabel("算法", fontsize=12)
        ax.set_ylabel("成功比例", fontsize=12)
        ax.set_title("算法鲁棒性 (成功运行比例)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 在柱子上添加数值
        for bar, val in zip(bars, robustness):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    
    # 5. 算法综合评分
    ax = axes[4]
    if analysis["summary"]:
        algorithms = list(analysis["summary"].keys())
        
        # 计算综合评分（归一化）
        scores = []
        for algo in algorithms:
            summary = analysis["summary"][algo]
            # 归一化：误差越小越好，相似度越高越好，时间越短越好，鲁棒性越高越好
            norm_error = 1 - (summary["avg_reward_diff"] / max(1, summary["avg_reward_diff"] * 10))
            norm_similarity = summary["avg_policy_similarity"]
            norm_time = 1 - (summary["avg_time"] / max(1, summary["avg_time"] * 10))
            norm_robustness = summary["robustness"]
            
            # 加权平均
            composite_score = (norm_error * 0.3 + norm_similarity * 0.3 + 
                             norm_time * 0.2 + norm_robustness * 0.2)
            scores.append(composite_score)
        
        bars = ax.bar(algorithms, scores, color=['blue', 'green', 'red'])
        ax.set_xlabel("算法", fontsize=12)
        ax.set_ylabel("综合评分", fontsize=12)
        ax.set_title("算法综合评分", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. 噪声影响总结
    ax = axes[5]
    ax.axis('off')
    
    if analysis["best_algorithms"]:
        text_content = "算法推荐总结:\n\n"
        for category, data in analysis["best_algorithms"].items():
            algo = data["algorithm"]
            metric_name = {
                "reward_recovery": "奖励恢复",
                "policy_similarity": "策略相似度",
                "efficiency": "计算效率",
                "robustness": "噪声鲁棒性",
            }.get(category, category)
            
            metric_value = list(data.values())[1]
            if category == "reward_recovery":
                text_content += f"• {metric_name}: {algo} (误差: {metric_value:.4f})\n"
            elif category == "policy_similarity":
                text_content += f"• {metric_name}: {algo} (相似度: {metric_value:.2%})\n"
            elif category == "efficiency":
                text_content += f"• {metric_name}: {algo} (时间: {metric_value:.2f}秒)\n"
            elif category == "robustness":
                text_content += f"• {metric_name}: {algo} (成功率: {metric_value:.2%})\n"
        
        ax.text(0.1, 0.5, text_content, fontsize=12, verticalalignment='center',
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("逆强化学习算法综合实验分析", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig_path = f"{save_dir}/experiment_results_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {fig_path}")
    
    plt.show()


def generate_report(results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """
    生成实验报告
    
    :param results: 实验结果
    :param analysis: 分析结果
    :return: 报告字符串
    """
    report = []
    report.append("=" * 80)
    report.append("逆强化学习算法综合实验报告")
    report.append("=" * 80)
    report.append("")
    
    # 实验配置
    config = results["config"]
    report.append("实验配置:")
    report.append(f"  环境网格大小: {config['env_config'].get('grid_size', '默认')}")
    report.append(f"  专家轨迹数量: {config['n_trajectories']}")
    report.append(f"  噪声水平测试: {config['noise_levels']}")
    report.append(f"  折扣因子: {config['gamma']}")
    report.append(f"  评估回合数: {config['n_episodes_eval']}")
    report.append(f"  随机种子: {config['seed']}")
    report.append("")
    
    # 实验结果摘要
    report.append("实验结果摘要:")
    if analysis["summary"]:
        for algo, summary in analysis["summary"].items():
            report.append(f"  {algo}:")
            report.append(f"    平均奖励恢复误差: {summary['avg_reward_diff']:.4f}")
            report.append(f"    平均奖励相关性: {summary['avg_reward_corr']:.4f}")
            report.append(f"    平均策略相似度: {summary['avg_policy_similarity']:.2%}")
            report.append(f"    平均计算时间: {summary['avg_time']:.2f}秒")
            report.append(f"    鲁棒性(成功比例): {summary['robustness']:.2%}")
            report.append("")
    
    # 算法推荐
    report.append("算法推荐:")
    if analysis["best_algorithms"]:
        for category, data in analysis["best_algorithms"].items():
            algo = data["algorithm"]
            metric_value = list(data.values())[1]
            
            if category == "reward_recovery":
                report.append(f"  • 最佳奖励恢复算法: {algo} (平均误差: {metric_value:.4f})")
            elif category == "policy_similarity":
                report.append(f"  • 最佳策略相似度算法: {algo} (平均相似度: {metric_value:.2%})")
            elif category == "efficiency":
                report.append(f"  • 最有效率算法: {algo} (平均时间: {metric_value:.2f}秒)")
            elif category == "robustness":
                report.append(f"  • 最鲁棒算法: {algo} (成功率: {metric_value:.2%})")
        report.append("")
    
    # 噪声影响分析
    report.append("噪声影响分析:")
    for noise, data in analysis.get("noise_impact", {}).items():
        report.append(f"  噪声水平 {noise}:")
        for algo, metrics in data.items():
            if metrics:
                reward_diff = metrics.get('reward_diff')
                policy_similarity = metrics.get('policy_similarity')
                reward_diff_str = f"{reward_diff:.4f}" if reward_diff is not None else "N/A"
                policy_sim_str = f"{policy_similarity:.2%}" if policy_similarity is not None else "N/A"
                report.append(f"    {algo}: 奖励误差={reward_diff_str}, 策略相似度={policy_sim_str}")
        report.append("")
    
    # 建议和洞察
    report.append("建议和洞察:")
    if analysis["recommendations"]:
        for i, rec in enumerate(analysis["recommendations"], 1):
            report.append(f"  {i}. {rec}")
    else:
        report.append("  无具体建议（实验数据不足）")
    
    report.append("")
    report.append("=" * 80)
    report.append("实验完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """主函数：运行完整实验并生成报告"""
    print("=" * 80)
    print("逆强化学习算法综合实验")
    print("=" * 80)
    
    # 运行实验（使用较小网格以加快速度）
    print("\n开始运行实验...")
    start_time = time.time()
    
    results = run_experiment(
        env_config={"grid_size": 8, "seed": 42},
        n_trajectories=50,
        noise_levels=[0.0, 0.1, 0.2, 0.3],
        gamma=0.99,
        n_episodes_eval=100,
        seed=42,
    )
    
    elapsed = time.time() - start_time
    print(f"\n实验完成，总耗时: {elapsed:.2f}秒")
    
    # 分析结果
    print("\n分析实验结果...")
    analysis = analyze_results(results)
    
    # 生成报告
    print("\n生成实验报告...")
    report = generate_report(results, analysis)
    print("\n" + report)
    
    # 保存结果到文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_results_{timestamp}.json"
    
    # 转换numpy数组为列表以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    serializable_results = json.loads(json.dumps(results, default=convert_to_serializable))
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验结果已保存至: {results_file}")
    
    # 保存报告到文件
    report_file = f"experiment_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"实验报告已保存至: {report_file}")
    
    # 可视化结果
    print("\n生成可视化图表...")
    try:
        visualize_results(results, analysis, save_dir=".")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n实验流程完成！")


if __name__ == "__main__":
    main()