"""
深度特征提取器模块（Deep Feature Extractor）
用于逆强化学习的深度特征版本

使用PyTorch实现神经网络，以智能体周围的局部网格作为输入，
预测状态的奖励值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LocalGridRewardNet(nn.Module):
    """
    基于局部网格的奖励预测神经网络
    
    输入：智能体周围 window_size x window_size 的局部网格
    输出：该状态的奖励值（标量）
    
    设计理念：
    - 使用局部网格而非全局坐标，避免简单记忆位置
    - 学习空间模式和地形分布特征
    - 支持泛化到不同地图配置
    """
    
    def __init__(
        self,
        window_size: int = 5,
        n_terrain_types: int = 5,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.2,
        use_cnn: bool = True,
    ):
        """
        初始化局部网格奖励网络
        
        :param window_size: 局部网格窗口大小（奇数，如3,5,7）
        :param n_terrain_types: 地形类型数量（默认5种）
        :param hidden_dims: 隐藏层维度
        :param dropout_rate: dropout比率，防止过拟合
        :param use_cnn: 是否使用CNN处理空间特征（推荐True）
        """
        super().__init__()
        self.window_size = window_size
        self.n_terrain_types = n_terrain_types
        self.use_cnn = use_cnn
        
        if use_cnn:
            # CNN架构处理空间特征
            # 输入形状: (batch_size, n_terrain_types, window_size, window_size)
            # 使用one-hot编码的地形类型
            
            # 计算CNN输出维度
            cnn_channels = [n_terrain_types, 32, 64]
            self.conv_layers = nn.ModuleList()
            
            # 卷积层
            self.conv_layers.append(
                nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
            )
            self.conv_layers.append(
                nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
            )
            
            # 池化层
            self.pool = nn.MaxPool2d(2)
            
            # 计算卷积后特征图大小
            conv_output_size = window_size // 2  # 经过一次池化
            cnn_output_dim = cnn_channels[2] * conv_output_size * conv_output_size
            
            # 全连接层
            fc_input_dim = cnn_output_dim
        else:
            # 全连接网络（扁平化局部网格）
            fc_input_dim = window_size * window_size * n_terrain_types
        
        # 构建全连接层
        fc_layers = []
        prev_dim = fc_input_dim
        
        for hidden_dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层：单个奖励值
        fc_layers.append(nn.Linear(prev_dim, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, local_grid: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        :param local_grid: 局部网格张量
                        形状: (batch_size, window_size, window_size) 或
                              (batch_size, window_size, window_size, n_terrain_types)
                        值: 地形类型索引（0-4）或one-hot编码
        :return: 奖励值，形状 (batch_size,)
        """
        batch_size = local_grid.shape[0]
        
        # 确保输入是4D张量 (batch_size, channels, height, width)
        if local_grid.dim() == 3:
            # 输入是地形类型索引，转换为one-hot编码
            local_grid = local_grid.long()
            one_hot = torch.zeros(
                batch_size, self.n_terrain_types, self.window_size, self.window_size,
                device=local_grid.device
            )
            local_grid = one_hot.scatter_(1, local_grid.unsqueeze(1), 1)
        elif local_grid.dim() == 4:
            # 输入已经是one-hot编码，调整维度顺序
            # 从 (batch_size, height, width, channels) 到 (batch_size, channels, height, width)
            if local_grid.shape[-1] == self.n_terrain_types:
                local_grid = local_grid.permute(0, 3, 1, 2).contiguous()
        
        if self.use_cnn:
            # CNN处理
            x = local_grid
            for conv_layer in self.conv_layers:
                x = F.relu(conv_layer(x))
            x = self.pool(x)
            x = x.view(batch_size, -1)  # 扁平化
        else:
            # 全连接网络处理
            x = local_grid.view(batch_size, -1)
        
        # 全连接层
        rewards = self.fc_layers(x).squeeze(-1)  # 去掉最后一个维度
        
        return rewards
    
    def predict_reward_matrix(self, env, device: Optional[str] = None) -> np.ndarray:
        """
        预测整个环境的奖励矩阵
        
        :param env: AdvancedGridWorld环境
        :param device: 计算设备（'cpu' 或 'cuda'）
        :return: 奖励矩阵，形状 (grid_size, grid_size)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.eval()
        reward_matrix = np.zeros((env.grid_size, env.grid_size))
        
        with torch.no_grad():
            for x in range(env.grid_size):
                for y in range(env.grid_size):
                    # 获取局部网格
                    local_grid = self._get_local_grid(env, (x, y))
                    local_grid_tensor = torch.tensor(
                        local_grid, dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    
                    # 预测奖励
                    reward = self(local_grid_tensor).cpu().item()
                    reward_matrix[x, y] = reward
        
        return reward_matrix
    
    def _get_local_grid(
        self, env, center: Tuple[int, int], fill_value: int = 1
    ) -> np.ndarray:
        """
        获取以center为中心的局部网格
        
        :param env: 环境对象
        :param center: 中心坐标 (x, y)
        :param fill_value: 边界填充值（默认1=障碍物）
        :return: 局部网格，形状 (window_size, window_size)
        """
        half = self.window_size // 2
        grid = np.full((self.window_size, self.window_size), fill_value, dtype=int)
        
        center_x, center_y = center
        
        for i in range(self.window_size):
            for j in range(self.window_size):
                map_x = center_x + i - half
                map_y = center_y + j - half
                
                if 0 <= map_x < env.grid_size and 0 <= map_y < env.grid_size:
                    grid[i, j] = env.grid[map_x, map_y]
        
        return grid


class DeepFeatureExtractor:
    """
    深度特征提取器包装类
    
    提供与原始线性特征提取器兼容的接口，
    便于集成到现有IRL算法中。
    """
    
    def __init__(
        self,
        network: Optional[nn.Module] = None,
        window_size: int = 5,
        device: Optional[str] = None,
    ):
        """
        初始化深度特征提取器
        
        :param network: 预训练的神经网络（如为None则创建默认网络）
        :param window_size: 局部网格窗口大小
        :param device: 计算设备
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if network is None:
            self.network = LocalGridRewardNet(window_size=window_size).to(device)
        else:
            self.network = network.to(device)
        
        self.window_size = window_size
    
    def get_features(self, env, state: Tuple[int, int]) -> torch.Tensor:
        """
        获取状态的深度特征表示（神经网络隐藏层激活）
        
        :param env: 环境对象
        :param state: 状态坐标 (x, y)
        :return: 深度特征张量
        """
        self.network.eval()
        with torch.no_grad():
            local_grid = self._get_local_grid_tensor(env, state)
            # 提取隐藏层激活（这里简化处理，实际可能需要修改网络结构）
            return local_grid
    
    def get_reward(self, env, state: Tuple[int, int]) -> float:
        """
        预测状态的奖励值
        
        :param env: 环境对象
        :param state: 状态坐标 (x, y)
        :return: 预测的奖励值
        """
        self.network.eval()
        with torch.no_grad():
            local_grid = self._get_local_grid_tensor(env, state)
            reward = self.network(local_grid).cpu().item()
            return reward
    
    def get_reward_matrix(self, env) -> np.ndarray:
        """
        获取整个环境的奖励矩阵
        
        :param env: 环境对象
        :return: 奖励矩阵，形状 (grid_size, grid_size)
        """
        return self.network.predict_reward_matrix(env, self.device)
    
    def _get_local_grid_tensor(self, env, center: Tuple[int, int]) -> torch.Tensor:
        """获取局部网格张量"""
        local_grid = self.network._get_local_grid(env, center)
        return torch.tensor(local_grid, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'window_size': self.window_size,
            'network_config': {
                'window_size': self.network.window_size,
                'n_terrain_types': self.network.n_terrain_types,
                'use_cnn': self.network.use_cnn,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        
        # 重建网络
        config = checkpoint['network_config']
        network = LocalGridRewardNet(
            window_size=config['window_size'],
            n_terrain_types=config['n_terrain_types'],
            use_cnn=config['use_cnn'],
        )
        network.load_state_dict(checkpoint['network_state_dict'])
        
        # 创建提取器
        extractor = cls(network=network, window_size=checkpoint['window_size'])
        return extractor


def test_deep_feature_extractor():
    """测试深度特征提取器"""
    from environment import AdvancedGridWorld
    
    print("测试深度特征提取器...")
    
    # 创建环境
    env = AdvancedGridWorld()
    
    # 创建深度特征提取器
    extractor = DeepFeatureExtractor(window_size=5)
    
    # 测试单个状态
    test_state = (3, 4)
    reward = extractor.get_reward(env, test_state)
    print(f"状态 {test_state} 的预测奖励: {reward:.4f}")
    
    # 测试整个奖励矩阵
    reward_matrix = extractor.get_reward_matrix(env)
    print(f"奖励矩阵形状: {reward_matrix.shape}")
    print(f"奖励范围: [{reward_matrix.min():.4f}, {reward_matrix.max():.4f}]")
    
    # 测试保存和加载
    extractor.save("test_deep_feature.pth")
    loaded_extractor = DeepFeatureExtractor.load("test_deep_feature.pth")
    
    # 验证加载的模型
    reward_loaded = loaded_extractor.get_reward(env, test_state)
    print(f"加载模型后状态 {test_state} 的预测奖励: {reward_loaded:.4f}")
    
    import os
    if os.path.exists("test_deep_feature.pth"):
        os.remove("test_deep_feature.pth")
    
    print("测试完成!")


if __name__ == "__main__":
    test_deep_feature_extractor()