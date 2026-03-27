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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 0. Random Maze Generator & Dummy Map
# ==========================================
# Use this exact dummy map
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
        
        # Exclude (1,1) so it perfectly matches the dummy map
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
# 1. Custom Combined Extractor
# ==========================================
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom Extractor for Dict observations.
    - 'observation': Processed by AlignedMazeCNN (64x64 -> 8x8)
    - 'achieved_goal' & 'desired_goal': Flattened
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == "observation":
                n_input_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                extractors[key] = cnn
                
                with torch.no_grad():
                    sample = torch.zeros(1, *subspace.shape)
                    cnn_out = cnn(sample).shape[1]
                total_concat_size += cnn_out
                
            else:
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
# 2. Dynamic Pixel Wrapper (HER + Map Fixes)
# ==========================================
class DynamicAlignedPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        self.img_size = 64 
        self.renderer = mujoco.Renderer(self.model, height=self.img_size, width=self.img_size)
        
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat = [0, 0, -10.0] 
        self.camera.distance = 20.0 
        self.camera.elevation = -90
        self.camera.azimuth = 90 
        
        new_spaces = env.observation_space.spaces.copy()
        new_spaces['observation'] = gym.spaces.Box(
            low=0, high=255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        self.renderer.update_scene(self.data, camera=self.camera)
        pixels = self.renderer.render()
        self._last_obs = pixels
        
        new_obs = obs.copy()
        new_obs['observation'] = pixels
        return new_obs
        
    def reset(self, **kwargs):
        # 1. Generate new map & update internal array
        new_maze = generate_single_random_maze()
        maze_obj = self.env.unwrapped.maze
        model = self.env.unwrapped.model
        maze_obj._maze_map = np.array(new_maze)
        
        # 2. Get all inner blocks
        geom_names = [model.names[model.name_geomadr[i]:].split(b'\x00')[0].decode('utf-8') for i in range(model.ngeom)]
        block_geoms = [name for name in geom_names if name.startswith("block_")]
        inner_block_geoms = [name for name in block_geoms if not (name.startswith("block_0_") or name.startswith("block_7_") or name.endswith("_0") or name.endswith("_7"))]
        
        # 3. THE GHOST PROTOCOL: Toggle visibility and collision
        for geom_name in inner_block_geoms:
            geom_id = model.geom(geom_name).id
            
            # Math to figure out exactly which grid cell this block is sitting on
            x = model.geom_pos[geom_id][0]
            y = model.geom_pos[geom_id][1]
            j = int(round((x + maze_obj.x_map_center) / maze_obj.maze_size_scaling - 0.5))
            i = int(round((maze_obj.y_map_center - y) / maze_obj.maze_size_scaling - 0.5))
            
            if new_maze[i][j] == 1:
                # WALL SHOULD EXIST: Bring to surface and make solid
                model.geom_pos[geom_id][2] = 0.0     
                model.geom_contype[geom_id] = 1      
                model.geom_conaffinity[geom_id] = 1  
            else:
                # WALL SHOULD NOT EXIST: Bury underground and turn off collision
                model.geom_pos[geom_id][2] = -10.0   
                model.geom_contype[geom_id] = 0      
                model.geom_conaffinity[geom_id] = 0  
                
        # 4. Update valid spawn locations
        empty_locations = []
        for i in range(maze_obj.map_length):
            for j in range(maze_obj.map_width):
                if new_maze[i][j] == 0:
                    x = (j + 0.5) * maze_obj.maze_size_scaling - maze_obj.x_map_center
                    y = maze_obj.y_map_center - (i + 0.5) * maze_obj.maze_size_scaling
                    empty_locations.append(np.array([x, y]))
                    
        maze_obj._unique_goal_locations = empty_locations.copy()
        maze_obj._unique_reset_locations = empty_locations.copy()
        maze_obj._combined_locations = empty_locations.copy()
        
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
                

    def compute_reward(self, achieved_goal, desired_goal, info):
        # HER FIX: VecFrameStack concatenates goals (e.g., shape becomes 8 instead of 2). 
        # We only want to compute reward based on the current timestep (last 2 coordinates).
        if achieved_goal.shape[-1] > 2:
            achieved_goal = achieved_goal[..., -2:]
            desired_goal = desired_goal[..., -2:]
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()
        super().close()
        
    def render(self):
        return self._last_obs

def make_wrapped_env(env_id, seed=0, log_dir=None):
    def _init():
        # Inject the dummy map here so MuJoCo compiles the correct number of blocks!
        env = gym.make(env_id, render_mode="rgb_array", maze_map=DUMMY_FULL_MAP)
        env = DynamicAlignedPixelWrapper(env)
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
    save_name = "sac_pixelpm_aligned_64_dynamic" 
    
    log_dir = f"logs/tb/{save_name}"
    eval_log_dir = f"./logs/{save_name}"
    checkpoint_dir = f"./logs/{save_name}/checkpoints"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training {save_name} with {num_envs} environments...")
    print(f"Using CustomCombinedExtractor (64x64 -> 8x8) and Dynamic Random Maps")

    # --- Setup Training Envs ---
    env_fns = [make_wrapped_env(env_id, seed=i, log_dir=log_dir) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='last')
    vec_env = VecTransposeImage(vec_env) 

    # --- Setup Eval Env ---
    eval_env = DummyVecEnv([make_wrapped_env(env_id, log_dir=eval_log_dir)])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='last')
    eval_env = VecTransposeImage(eval_env)

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
        # RAM FIX: if OOM crash, reduce buffer_size.
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