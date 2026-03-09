import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Assuming train.py contains AlignedPixelWrapper
from train import DynamicAlignedPixelWrapper

# --- 1. The Feature Hook ---
features_storage = []

def hook_fn(module, input, output):
    """Callback to grab the 64x8x8 tensor during forward pass"""
    features_storage.append(output.detach().cpu().numpy())

# --- 2. Setup Environment & Model ---
env_id = "PointMaze_Medium-v3"
save_name = "sac_pixelpm_aligned_64_dynamic"
model_path = f"./logs/{save_name}/best_model.zip"

def make_env():
    env = gym.make(env_id, render_mode="rgb_array")
    return DynamicAlignedPixelWrapper(env)

eval_env = DummyVecEnv([make_env])
eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='last')
eval_env = VecTransposeImage(eval_env)

# Get the base environment to access Ground Truth positions
base_env = eval_env.envs[0].unwrapped
maze_array = np.array(base_env.maze.maze_map)

print(f"Maze Map: {maze_array}")
breakpoint()
# --- Helper: Convert Continuous XY to Grid Indices ---
def get_agent_grid_pos(env):
    """
    Retrieves the agent's (x, y) and converts it to (row, col) indices.
    """
    # 1. Get continuous position (MuJoCo specific)
    # 'data.qpos' is standard for MuJoCo envs. 
    # Check if your specific env uses 'env.point.pos' instead.
    if hasattr(env, 'point'):
        x, y = env.point.pos[:2]
    else:
        x, y = env.data.qpos[:2]

    # 2. Convert to Grid Indices using the Maze utility
    # Gymnasium Robotics Maze objects usually have this helper method
    row, col = env.maze.cell_xy_to_rowcol([x, y])
    # print(f"Agent Continuous Pos: ({x:.2f}, {y:.2f}) -> Grid Pos: (Row: {row}, Col: {col})")
    return np.array([row, col])

# Load Model
model = SAC.load(model_path, env=eval_env)

# Register Hook
target_layer = model.policy.actor.features_extractor.extractors['observation'][3]
handle = target_layer.register_forward_hook(hook_fn)

# --- 3. Data Collection Loop ---
all_episode_features = []
all_current_pos = []  # Stores (row, col) before step
all_next_pos = []     # Stores (row, col) after step

print(f"Collecting data for 10 episodes (100 steps each)...")

for ep in range(100):
    obs = eval_env.reset()
    
    ep_features = []
    ep_curr_pos = []
    ep_next_pos = []
    
    for step in range(100):
        features_storage.clear()
        
        # 1. Record CURRENT Position (Before moving)
        # We must use base_env because 'obs' is just pixels
        curr_grid = get_agent_grid_pos(base_env)
        ep_curr_pos.append(curr_grid)
        
        # 2. Forward pass & Step
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        
        # 3. Record NEXT Position (After moving)
        next_grid = get_agent_grid_pos(base_env)
        ep_next_pos.append(next_grid)
        
        # Store features
        if features_storage:
            ep_features.append(features_storage[0][0]) 
            
        if dones[0]:
            break
    
    # Append episode data
    all_episode_features.append(np.array(ep_features))
    all_current_pos.append(np.array(ep_curr_pos))
    all_next_pos.append(np.array(ep_next_pos))
    
    print(f"Finished Episode {ep+1}, steps: {len(ep_features)}")

# Cleanup
handle.remove()

# --- 4. Formatting Data ---
# Convert lists to arrays
final_features = np.array(all_episode_features) # (Episodes, Steps, 64, 8, 8)
final_curr_pos = np.array(all_current_pos)      # (Episodes, Steps, 2)
final_next_pos = np.array(all_next_pos)         # (Episodes, Steps, 2)

print("\n" + "="*30)
print(f"Collection Complete!")
print(f"Features Shape: {final_features.shape}")
print(f"Current Pos Labels Shape: {final_curr_pos.shape}") # e.g. (10, 100, 2)
print(f"Next Pos Labels Shape:    {final_next_pos.shape}")
print("="*30)

# --- 5. Saving Data ---
save_path = "maze_probing_data_labeled.npz"
np.savez(save_path, 
         features=final_features,   # The CNN activations
         maze_map=maze_array,       # The static Wall/Path map
         agent_pos_curr=final_curr_pos, # Label: Where agent IS
         agent_pos_next=final_next_pos  # Label: Where agent IS GOING
         )

print(f"Data saved to {save_path}")