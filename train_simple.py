import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import gymnasium_robotics
import mujoco

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 0. Random Maze Generator & Dummy Map
# ==========================================
DUMMY_FULL_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

def generate_single_random_maze():
    while True:
        maze = np.ones((8, 8), dtype=int)
        maze[1:-1, 1:-1] = 0  
        
        inner_indices = [(r, c) for r in range(1, 7) for c in range(1, 7) if (r, c) != (1, 1)]
        wall_indices = random.sample(inner_indices, 9)
        for r, c in wall_indices:
            maze[r, c] = 1
            
        zeros = np.argwhere(maze == 0)
        if len(zeros) != 27: continue
            
        start_node = tuple(zeros[0])
        visited = set()
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if maze[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    
        if len(visited) == 27:
            return maze.tolist()

# ==========================================
# 1. Custom Combined Extractor (Simplified for Matrix)
# ==========================================
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    因为我们将图片换成了 8x8 的矩阵，不再需要 CNN。
    直接将全部输入展平 (Flatten) 拼接后交给 SAC 的全连接层 (MLP) 处理。
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            extractors[key] = nn.Flatten()
            total_concat_size += np.prod(subspace.shape)
        
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

# ==========================================
# 2. Dynamic Wall Matrix Wrapper 
# ==========================================
class DynamicWallMatrixWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        # 将 observation 空间替换为 8x8 的矩阵 (值范围 0~1)
        new_spaces = env.observation_space.spaces.copy()
        new_spaces['observation'] = gym.spaces.Box(
            low=0, high=1, shape=(8, 8), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(new_spaces)
        
        self.solid_wall_geom_ids = set()
        self.current_maze_matrix = np.zeros((8, 8), dtype=np.float32)

    def observation(self, obs):
        new_obs = obs.copy()
        # 替换原有的 observation 为当前地图的 8x8 墙壁矩阵
        new_obs['observation'] = self.current_maze_matrix.copy()
        return new_obs
        
    def reset(self, **kwargs):
        new_maze = generate_single_random_maze()
        self.current_maze_matrix = np.array(new_maze, dtype=np.float32)
        
        maze_obj = self.env.unwrapped.maze
        model = self.env.unwrapped.model
        maze_obj._maze_map = self.current_maze_matrix
        
        self.solid_wall_geom_ids.clear()
        
        geom_names = [model.names[model.name_geomadr[i]:].split(b'\x00')[0].decode('utf-8') for i in range(model.ngeom)]
        block_geoms = [name for name in geom_names if name.startswith("block_")]
        
        for geom_name in block_geoms:
            geom_id = model.geom(geom_name).id
            is_inner = not (geom_name.startswith("block_0_") or geom_name.startswith("block_7_") or geom_name.endswith("_0") or geom_name.endswith("_7"))
            
            if is_inner:
                x = model.geom_pos[geom_id][0]
                y = model.geom_pos[geom_id][1]
                j = int(round((x + maze_obj.x_map_center) / maze_obj.maze_size_scaling - 0.5))
                i = int(round((maze_obj.y_map_center - y) / maze_obj.maze_size_scaling - 0.5))
                
                if new_maze[i][j] == 1:
                    model.geom_pos[geom_id][2] = 0.0     
                    model.geom_contype[geom_id] = 1      
                    model.geom_conaffinity[geom_id] = 1  
                    self.solid_wall_geom_ids.add(geom_id)
                else:
                    model.geom_pos[geom_id][2] = -10.0   
                    model.geom_contype[geom_id] = 0      
                    model.geom_conaffinity[geom_id] = 0  
            else:
                self.solid_wall_geom_ids.add(geom_id)
                
        empty_locations = []
        wall_locations = []
        for i in range(maze_obj.map_length):
            for j in range(maze_obj.map_width):
                x = (j + 0.5) * maze_obj.maze_size_scaling - maze_obj.x_map_center
                y = maze_obj.y_map_center - (i + 0.5) * maze_obj.maze_size_scaling
                if new_maze[i][j] == 0:
                    empty_locations.append(np.array([x, y]))
                else:
                    wall_locations.append(np.array([x, y]))
                    
        def point_to_segment_dist(w, a, b):
            ab = b - a
            l2 = np.sum(ab**2)
            if l2 == 0:
                return np.linalg.norm(w - a)
            t = np.dot(w - a, ab) / l2
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            return np.linalg.norm(w - proj)

        threshold = maze_obj.maze_size_scaling * 0.45 
        while True:
            start_idx = random.randint(0, len(empty_locations) - 1)
            goal_idx = random.randint(0, len(empty_locations) - 1)
            
            if start_idx == goal_idx:
                continue
                
            start_pos = empty_locations[start_idx]
            goal_pos = empty_locations[goal_idx]
            
            intersect = False
            for w in wall_locations:
                if point_to_segment_dist(w, start_pos, goal_pos) < threshold:
                    intersect = True
                    break
                    
            if intersect:
                break 
                
        maze_obj._unique_reset_locations = [start_pos]
        maze_obj._unique_goal_locations = [goal_pos]
        maze_obj._combined_locations = [start_pos, goal_pos]
        
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        touching_wall = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.solid_wall_geom_ids or contact.geom2 in self.solid_wall_geom_ids:
                touching_wall = True
                break
                
        if touching_wall:
            reward -= 0.4  
            
        return self.observation(obs), reward, terminated, truncated, info
                
    def compute_reward(self, achieved_goal, desired_goal, info):
        if achieved_goal.shape[-1] > 2:
            achieved_goal = achieved_goal[..., -2:]
            desired_goal = desired_goal[..., -2:]
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

def make_wrapped_env(env_id, seed=0, log_dir=None):
    def _init():
        env = gym.make(env_id, maze_map=DUMMY_FULL_MAP)
        env = DynamicWallMatrixWrapper(env)
        if log_dir:
            env = Monitor(env, log_dir)
        return env
    return _init

# ==========================================
# 3. Main Training Script
# ==========================================
if __name__ == "__main__":
    env_id = "PointMaze_Medium-v3"
    num_envs = 10  
    save_name = "sac_matrixpm_forced_bypass_touch_penalty04"
    
    log_dir = f"logs/tb/{save_name}"
    eval_log_dir = f"./logs/{save_name}"
    checkpoint_dir = f"./logs/{save_name}/checkpoints"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training {save_name} with {num_envs} environments...")
    print(f"Using Matrix Observation (8x8 -> Flat) and Dynamic Random Maps")

    # --- Setup Training Envs ---
    env_fns = [make_wrapped_env(env_id, seed=i, log_dir=log_dir) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    # 移除 VecFrameStack 和 VecTransposeImage，因为处理的是静态矩阵而非动态像素
    
    # --- Setup Eval Env ---
    eval_env = DummyVecEnv([make_wrapped_env(env_id, log_dir=eval_log_dir)])

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=5000 // num_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // num_envs, 1), 
        save_path=checkpoint_dir,
        name_prefix="sac_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    callback = CallbackList([checkpoint_callback, eval_callback])

    # --- Define Model ---
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000, 
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        ent_coef='auto',
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1, 
        policy_kwargs=dict(
            features_extractor_class=CustomCombinedExtractor, 
            features_extractor_kwargs=dict(),
            net_arch=[256, 256, 256], 
        ),
        tensorboard_log=log_dir
    )

    print("\nPolicy Architecture:")
    print(model.policy)
    print("-" * 30)
    
    
    print("Starting training loop...")
    model.learn(total_timesteps=3e6, callback=callback, log_interval=4)
    model.save(f"{eval_log_dir}/final_model")