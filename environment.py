"""
环境类：AdvancedGridWorld

=============================================
               游戏规则说明
=============================================

【核心目标】
1. 玩家从起点 (S) 出发，需要到达终点 (G)
2. 每走一步都会获得奖励/惩罚，目标是最大化累计奖励

【网格世界】
1. N×N 网格（默认10×10）
2. 坐标系统：(x,y)，x是行索引（0在最上），y是列索引（0在最左）

【地形类型】
1. 普通地面（TERRAIN_NORMAL=0）：-1点/步
2. 障碍物（TERRAIN_OBSTACLE=1）：不可通过，只能停留在原地
3. 泥潭（TERRAIN_MUD=2）：-5点/步（高惩罚区）
4. 草地（TERRAIN_GRASS=3）：0点/步（低惩罚区）
5. 终点（TERRAIN_GOAL=4）：+100点（游戏结束）

【动作空间】
0 = 上（↑）：(x-1, y)
1 = 下（↓）：(x+1, y)
2 = 左（←）：(x, y-1)
3 = 右（→）：(x, y+1)

【转移概率】
1. 目标动作执行概率：1 - transition_noise（默认0.8）
2. 随机噪声概率：transition_noise（默认0.2）
3. 噪声方向：与目标动作垂直的两个方向平分噪声概率
4. 遇到边界或障碍物：停留在原地

【奖励系统】
1. 立即奖励：取决于下一状态的地形类型
2. 到达终点：游戏结束，获得终点奖励
3. 每步惩罚：鼓励智能体尽快到达终点

【终止条件】
1. 到达终点（goal_state）
2. 超过最大步数（默认100步）

【特征表示】（用于IRL）
1. 地形类型指示特征（goal/obstacle/mud/grass）
2. 距离特征：到终点的归一化曼哈顿距离
3. 障碍物距离特征：到最近障碍物的归一化距离

【关键接口】
1. reset(): 重置环境到起点
2. step(action): 执行动作，返回(next_state, reward, done, info)
3. generate_trajectory(policy): 根据策略生成轨迹
4. 其他IRL专用接口详见类方法

【默认配置】
grid_size=10, transition_noise=0.2, obstacle_ratio=0.15,
mud_ratio=0.10, grass_ratio=0.10, max_steps=100
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


# ===================== 核心：Advanced GridWorld 环境类 =====================
class AdvancedGridWorld:
    def __init__(self, config: Optional[Dict] = None):
        """
        进阶版GridWorld环境初始化，所有参数可配置，完美适配IRL项目需求
        :param config: 环境配置字典，不传则使用默认配置
        """
        # 1. 默认配置（可直接修改这里的参数做实验）
        default_config = {
            "grid_size": 10,  # 网格大小 N x N
            "transition_noise": 0.2,  # 随机转移概率：80%执行目标动作，剩余平分到垂直方向
            "action_noise": 0.0,  # 专家动作噪声概率（用于消融实验）
            "obstacle_ratio": 0.15,  # 障碍物占比
            "mud_ratio": 0.10,  # 泥潭（高惩罚区）占比
            "grass_ratio": 0.10,  # 草地（低惩罚区）占比
            "reward_goal": 100.0,  # 终点奖励
            "reward_normal": -1.0,  # 普通地面每步奖励
            "reward_mud": -5.0,  # 泥潭每步惩罚
            "reward_grass": 0.0,  # 草地每步奖励
            "seed": 42,  # 随机种子，保证实验可复现
            "max_steps": 100,  # 单条轨迹最大步长
        }

        # 合并用户配置和默认配置
        self.config = default_config if config is None else {**default_config, **config}
        np.random.seed(self.config["seed"])

        # 2. 基础环境参数
        self.grid_size = self.config["grid_size"]
        self.n_states = self.grid_size * self.grid_size  # 总状态数（一维索引）
        self.n_actions = 4  # 动作空间：0=上, 1=下, 2=左, 3=右
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 动作对应的坐标变化

        # 3. 地形编码定义
        self.TERRAIN_NORMAL = 0  # 普通地面
        self.TERRAIN_OBSTACLE = 1  # 障碍物（不可穿过）
        self.TERRAIN_MUD = 2  # 泥潭（高惩罚）
        self.TERRAIN_GRASS = 3  # 草地（低惩罚）
        self.TERRAIN_GOAL = 4  # 终点（终止态）

        # 4. 初始化起点、终点（必须在 _generate_grid 之前，因为地图生成需要用到）
        self.start_state = (0, 0)  # 默认起点
        self.goal_state = (self.grid_size - 1, self.grid_size - 1)  # 默认终点

        # 5. 生成地图
        self.grid = self._generate_grid()
        self.grid[self.goal_state] = self.TERRAIN_GOAL

        # 6. 预计算核心矩阵（IRL算法必备）
        self.transition_matrix = (
            self._build_transition_matrix()
        )  # 转移概率矩阵 P[s][a][s']
        self.feature_matrix = (
            self._build_feature_matrix()
        )  # 全状态特征矩阵 (n_states, n_features)
        self.true_reward_matrix = (
            self._build_true_reward_matrix()
        )  # 真实奖励矩阵（ground truth）

        # 6. 运行时状态
        self.current_state = self.start_state
        self.current_step = 0

    # ===================== 内部工具方法：地图与矩阵初始化 =====================
    def _generate_grid(self) -> np.ndarray:
        """生成带多地形的网格地图，保证起点和终点无障碍物"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        total_cells = self.grid_size * self.grid_size

        # 随机生成障碍物
        n_obstacles = int(total_cells * self.config["obstacle_ratio"])
        obstacle_cells = []
        while len(obstacle_cells) < n_obstacles:
            x, y = np.random.randint(0, self.grid_size, 2)
            if (
                (x, y) != self.start_state
                and (x, y) != (self.grid_size - 1, self.grid_size - 1)
                and (x, y) not in obstacle_cells
            ):
                obstacle_cells.append((x, y))
        for x, y in obstacle_cells:
            grid[x, y] = self.TERRAIN_OBSTACLE

        # 随机生成泥潭
        n_mud = int(total_cells * self.config["mud_ratio"])
        mud_cells = []
        while len(mud_cells) < n_mud:
            x, y = np.random.randint(0, self.grid_size, 2)
            if (
                grid[x, y] == self.TERRAIN_NORMAL
                and (x, y) != self.start_state
                and (x, y) != (self.grid_size - 1, self.grid_size - 1)
            ):
                mud_cells.append((x, y))
        for x, y in mud_cells:
            grid[x, y] = self.TERRAIN_MUD

        # 随机生成草地
        n_grass = int(total_cells * self.config["grass_ratio"])
        grass_cells = []
        while len(grass_cells) < n_grass:
            x, y = np.random.randint(0, self.grid_size, 2)
            if (
                grid[x, y] == self.TERRAIN_NORMAL
                and (x, y) != self.start_state
                and (x, y) != (self.grid_size - 1, self.grid_size - 1)
            ):
                grass_cells.append((x, y))
        for x, y in grass_cells:
            grid[x, y] = self.TERRAIN_GRASS

        return grid

    def _build_transition_matrix(self) -> np.ndarray:
        """
        预构建转移概率矩阵（IRL算法核心依赖）
        维度：(n_states, n_actions, n_states)
        P[s][a][s'] = 从状态s执行动作a，转移到s'的概率
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        max_dist = 2 * (self.grid_size - 1)  # 地图最大曼哈顿距离

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                s = self._state_to_idx((x, y))
                # 终止态：100%留在自身
                if self.is_terminal((x, y)):
                    for a in range(self.n_actions):
                        P[s, a, s] = 1.0
                    continue
                # 障碍物：无意义，直接留自身
                if self.grid[x, y] == self.TERRAIN_OBSTACLE:
                    for a in range(self.n_actions):
                        P[s, a, s] = 1.0
                    continue

                # 非终止态：计算每个动作的转移概率
                for a in range(self.n_actions):
                    # 目标动作的方向
                    target_dx, target_dy = self.action_deltas[a]
                    # 垂直方向（用于随机噪声）
                    if a in [0, 1]:  # 上下动作，垂直方向是左右
                        noise_directions = [
                            self.action_deltas[2],
                            self.action_deltas[3],
                        ]
                    else:  # 左右动作，垂直方向是上下
                        noise_directions = [
                            self.action_deltas[0],
                            self.action_deltas[1],
                        ]

                    # 概率分配
                    prob_target = 1 - self.config["transition_noise"]
                    prob_noise = self.config["transition_noise"] / 2

                    # 1. 目标动作的转移
                    nx, ny = x + target_dx, y + target_dy
                    # 边界/障碍物检查：越界/撞墙则停在原地
                    if (
                        0 <= nx < self.grid_size
                        and 0 <= ny < self.grid_size
                        and self.grid[nx, ny] != self.TERRAIN_OBSTACLE
                    ):
                        next_s_target = self._state_to_idx((nx, ny))
                    else:
                        next_s_target = s
                    P[s, a, next_s_target] += prob_target

                    # 2. 噪声方向的转移
                    for dx, dy in noise_directions:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < self.grid_size
                            and 0 <= ny < self.grid_size
                            and self.grid[nx, ny] != self.TERRAIN_OBSTACLE
                        ):
                            next_s_noise = self._state_to_idx((nx, ny))
                        else:
                            next_s_noise = s
                        P[s, a, next_s_noise] += prob_noise
        return P

    def _get_single_feature(self, state: Tuple[int, int]) -> np.ndarray:
        """计算单个状态的特征向量φ(s)，所有连续特征归一化到[0,1]"""
        x, y = state
        max_dist = 2 * (self.grid_size - 1)  # 地图最大曼哈顿距离

        # ========== 核心特征（必选，6维，适配线性IRL） ==========
        # 1. 二元指示特征
        is_goal = 1.0 if self.grid[x, y] == self.TERRAIN_GOAL else 0.0
        is_obstacle = 1.0 if self.grid[x, y] == self.TERRAIN_OBSTACLE else 0.0
        is_mud = 1.0 if self.grid[x, y] == self.TERRAIN_MUD else 0.0
        is_grass = 1.0 if self.grid[x, y] == self.TERRAIN_GRASS else 0.0

        # 2. 距离特征（归一化）
        dist_to_goal = abs(x - self.goal_state[0]) + abs(y - self.goal_state[1])
        norm_dist_to_goal = 1 - (dist_to_goal / max_dist)  # 离终点越近，值越大

        # 到最近障碍物的距离
        obstacle_positions = np.argwhere(self.grid == self.TERRAIN_OBSTACLE)
        if len(obstacle_positions) == 0:
            dist_to_obstacle = max_dist
        else:
            dist_to_obstacle = np.min(
                [abs(x - ox) + abs(y - oy) for (ox, oy) in obstacle_positions]
            )
        norm_dist_to_obstacle = dist_to_obstacle / max_dist

        # 拼接特征向量
        feature = np.array(
            [
                is_goal,
                is_obstacle,
                is_mud,
                is_grass,
                norm_dist_to_goal,
                norm_dist_to_obstacle,
            ]
        )

        return feature

    def _build_feature_matrix(self) -> np.ndarray:
        """构建全状态特征矩阵 (n_states, n_features)，IRL优化直接调用"""
        feature_matrix = np.zeros(
            (self.n_states, self._get_single_feature((0, 0)).shape[0])
        )
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                s = self._state_to_idx((x, y))
                feature_matrix[s] = self._get_single_feature((x, y))
        return feature_matrix

    def _build_true_reward_matrix(self) -> np.ndarray:
        """构建真实奖励矩阵（ground truth），用于后续IRL恢复效果对比"""
        reward_matrix = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                terrain = self.grid[x, y]
                if terrain == self.TERRAIN_GOAL:
                    reward_matrix[x, y] = self.config["reward_goal"]
                elif terrain == self.TERRAIN_MUD:
                    reward_matrix[x, y] = self.config["reward_mud"]
                elif terrain == self.TERRAIN_GRASS:
                    reward_matrix[x, y] = self.config["reward_grass"]
                elif terrain == self.TERRAIN_NORMAL:
                    reward_matrix[x, y] = self.config["reward_normal"]
                else:  # 障碍物无奖励
                    reward_matrix[x, y] = 0.0
        return reward_matrix

    # ===================== 状态索引转换工具（一维/二维互转） =====================
    def _state_to_idx(self, state: Tuple[int, int]) -> int:
        """二维坐标(x,y)转一维状态索引"""
        x, y = state
        return x * self.grid_size + y

    def _idx_to_state(self, idx: int) -> Tuple[int, int]:
        """一维状态索引转二维坐标(x,y)"""
        x = idx // self.grid_size
        y = idx % self.grid_size
        return (x, y)

    # ===================== 标准RL环境核心接口 =====================
    def reset(self, start_state: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """重置环境，每局游戏开始调用"""
        if start_state is not None:
            self.current_state = start_state
        else:
            self.current_state = self.start_state
        self.current_step = 0
        return self.current_state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        执行动作，环境核心逻辑
        :param action: 0=上,1=下,2=左,3=右
        :return: next_state, reward, done, info
        """
        self.current_step += 1
        x, y = self.current_state
        s = self._state_to_idx((x, y))

        # 按转移概率采样下一状态
        next_s_probs = self.transition_matrix[s, action]
        next_s = np.random.choice(self.n_states, p=next_s_probs)
        next_state = self._idx_to_state(next_s)

        # 获取奖励
        reward = self.true_reward_matrix[next_state]

        # 判断是否终止
        done = self.is_terminal(next_state) or (
            self.current_step >= self.config["max_steps"]
        )

        # 额外信息（调试用）
        info = {
            "current_step": self.current_step,
            "feature": self._get_single_feature(next_state),
            "true_reward": reward,
        }

        # 更新当前状态
        self.current_state = next_state
        return next_state, reward, done, info

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """判断状态是否为终止态（终点）"""
        return state == self.goal_state

    # ===================== IRL专属接口（Student2核心依赖） =====================
    def get_transition_prob(
        self, state: Tuple[int, int], action: int
    ) -> Dict[Tuple[int, int], float]:
        """获取给定状态+动作下，所有下一状态的转移概率"""
        s = self._state_to_idx(state)
        next_s_probs = self.transition_matrix[s, action]
        prob_dict = {}
        for next_s in range(self.n_states):
            if next_s_probs[next_s] > 0:
                prob_dict[self._idx_to_state(next_s)] = next_s_probs[next_s]
        return prob_dict

    def get_features(self, state: Tuple[int, int]) -> np.ndarray:
        """获取单个状态的特征向量"""
        return self._get_single_feature(state)

    def get_feature_matrix(self) -> np.ndarray:
        """获取全状态特征矩阵"""
        return self.feature_matrix

    def get_true_reward(
        self, state: Optional[Tuple[int, int]] = None
    ) -> float | np.ndarray:
        """获取真实奖励，不传state则返回全地图奖励矩阵"""
        if state is None:
            return self.true_reward_matrix
        else:
            return self.true_reward_matrix[state]

    # ===================== 数据生成接口（Student1核心交付物） =====================
    def generate_trajectory(
        self, policy: np.ndarray, noise_level: Optional[float] = None
    ) -> Dict:
        """
        输入策略，生成单条完整轨迹
        :param policy: 策略矩阵 (n_states, n_actions)，policy[s][a] = 状态s选动作a的概率
        :param noise_level: 动作噪声概率，不传则用环境默认值
        :return: 轨迹字典，包含完整的状态、动作、奖励序列
        """
        if noise_level is None:
            noise_level = self.config["action_noise"]

        state = self.reset()
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        done = False

        while not done:
            s = self._state_to_idx(state)
            # 动作噪声：有概率随机选动作
            if np.random.rand() < noise_level:
                action = np.random.choice(self.n_actions)
            else:
                action = np.random.choice(self.n_actions, p=policy[s])

            next_state, reward, done, info = self.step(action)

            # 记录轨迹
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["next_states"].append(next_state)
            trajectory["dones"].append(done)

            state = next_state

        return trajectory

    def generate_expert_dataset(
        self,
        expert_policy: np.ndarray,
        n_trajectories: int,
        noise_level: Optional[float] = None,
    ) -> List[Dict]:
        """
        批量生成专家演示数据集（Student1核心交付物）
        :param expert_policy: 专家最优策略
        :param n_trajectories: 轨迹数量
        :param noise_level: 动作噪声等级（用于消融实验）
        :return: 轨迹列表，每个元素是单条轨迹字典
        """
        dataset = []
        for _ in range(n_trajectories):
            traj = self.generate_trajectory(expert_policy, noise_level)
            dataset.append(traj)
        return dataset

    def compare_trajectories(
        self, traj_a: Dict, traj_b: Dict
    ) -> Tuple[int, float, float]:
        """
        对比两条轨迹的优劣，生成偏好标签（适配Bradley-Terry模型）
        :return: label(1=A更好, 0=B更好, 0.5=平局), return_a, return_b
        """
        return_a = np.sum(traj_a["rewards"])
        return_b = np.sum(traj_b["rewards"])

        if return_a > return_b:
            return 1, return_a, return_b
        elif return_b > return_a:
            return 0, return_a, return_b
        else:
            return 0.5, return_a, return_b

    def generate_preference_dataset(
        self,
        expert_policy: np.ndarray,
        n_pairs: int,
        noise_level: Optional[float] = None,
    ) -> List[Tuple[Dict, Dict, int]]:
        """
        批量生成偏好对比数据集（Student1核心交付物，适配偏好学习）
        :param expert_policy: 专家策略
        :param n_pairs: 偏好对数量
        :param noise_level: 动作噪声
        :return: 偏好对列表 [(traj_a, traj_b, label)]
        """
        # 先生成足够多的轨迹
        n_trajectories = n_pairs * 2
        traj_pool = self.generate_expert_dataset(
            expert_policy, n_trajectories, noise_level
        )

        # 两两配对生成偏好数据
        preference_dataset = []
        for i in range(n_pairs):
            traj_a = traj_pool[2 * i]
            traj_b = traj_pool[2 * i + 1]
            label, _, _ = self.compare_trajectories(traj_a, traj_b)
            preference_dataset.append((traj_a, traj_b, label))

        return preference_dataset

    # ===================== 可视化接口（项目交付物+调试） =====================
    def plot_reward_heatmap(
        self,
        reward_matrix: Optional[np.ndarray] = None,
        title: str = "Reward Heatmap",
        save_path: Optional[str] = None,
    ):
        """绘制奖励热力图，支持真实奖励和IRL恢复的奖励对比"""
        if reward_matrix is None:
            reward_matrix = self.true_reward_matrix

        plt.figure(figsize=(8, 6))
        im = plt.imshow(reward_matrix, cmap="RdYlGn", origin="upper")
        plt.colorbar(im, label="Reward Value")
        plt.title(title, fontsize=14)
        plt.xlabel("Y Coordinate", fontsize=12)
        plt.ylabel("X Coordinate", fontsize=12)
        plt.grid(alpha=0.3)

        # 标注终点、起点、障碍物
        plt.text(
            self.goal_state[1],
            self.goal_state[0],
            "G",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )
        plt.text(
            self.start_state[1],
            self.start_state[0],
            "S",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )

        obstacle_pos = np.argwhere(self.grid == self.TERRAIN_OBSTACLE)
        for x, y in obstacle_pos:
            plt.text(y, x, "X", ha="center", va="center", color="black", fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_trajectory(
        self,
        trajectory: Dict,
        title: str = "Expert Trajectory",
        save_path: Optional[str] = None,
    ):
        """可视化轨迹在网格中的路径"""
        plt.figure(figsize=(8, 8))
        # 绘制网格背景
        plt.imshow(self.grid, cmap="Set3", origin="upper", vmin=0, vmax=4)
        plt.title(title, fontsize=14)
        plt.xlabel("Y Coordinate", fontsize=12)
        plt.ylabel("X Coordinate", fontsize=12)
        plt.grid(which="both", color="black", linestyle="-", linewidth=1)
        plt.xticks(
            np.arange(-0.5, self.grid_size, 1), np.arange(0, self.grid_size + 1, 1)
        )
        plt.yticks(
            np.arange(-0.5, self.grid_size, 1), np.arange(0, self.grid_size + 1, 1)
        )

        # 标注关键位置
        plt.text(
            self.goal_state[1],
            self.goal_state[0],
            "G",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )
        plt.text(
            self.start_state[1],
            self.start_state[0],
            "S",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )

        # 绘制轨迹路径
        states = trajectory["states"]
        x_coords = [s[0] for s in states]
        y_coords = [s[1] for s in states]
        plt.plot(
            y_coords,
            x_coords,
            color="red",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Trajectory",
        )
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# ===================== 工具函数：价值迭代（生成专家最优策略） =====================
def value_iteration(
    env: AdvancedGridWorld,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    价值迭代算法，求解MDP的最优策略和最优价值函数
    :param env: GridWorld环境
    :param gamma: 折扣因子
    :param theta: 收敛阈值
    :param max_iter: 最大迭代次数
    :return: 最优策略 (n_states, n_actions)，最优价值函数 (n_states,)
    """
    n_states = env.n_states
    n_actions = env.n_actions
    P = env.transition_matrix
    R = env.true_reward_matrix.flatten()  # 一维奖励向量

    # 初始化价值函数
    V = np.zeros(n_states)

    # 价值迭代主循环
    for i in range(max_iter):
        delta = 0
        # 对每个状态更新价值
        for s in range(n_states):
            old_v = V[s]
            # 计算每个动作的Q值
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = np.sum(P[s, a] * (R + gamma * V))
            # 最优价值是最大Q值
            V[s] = np.max(q_values)
            delta = max(delta, abs(old_v - V[s]))
        # 收敛判断
        if delta < theta:
            print(f"价值迭代在第{i + 1}轮收敛")
            break

    # 提取最优策略
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            q_values[a] = np.sum(P[s, a] * (R + gamma * V))
        # 最优动作：Q值最大的动作
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0  # 确定性策略

    return policy, V
