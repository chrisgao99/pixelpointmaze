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

# Get the base environment
base_env = eval_env.envs[0].unwrapped

def get_agent_grid_pos(env):
    """Retrieves the agent's (x, y) and converts it to (row, col) indices."""
    if hasattr(env, 'point'):
        x, y = env.point.pos[:2]
    else:
        x, y = env.data.qpos[:2]
    row, col = env.maze.cell_xy_to_rowcol([x, y])
    return np.array([row, col])

# Load Model
model = SAC.load(model_path, env=eval_env)

# Register Hook
target_layer = model.policy.actor.features_extractor.extractors['observation'][3]
handle = target_layer.register_forward_hook(hook_fn)

# --- 3. Data Collection Loop ---
all_episode_features = []
all_current_pos = []  
all_next_pos = []     
all_maze_maps = []    # NEW: Store the dynamic maps per episode

print(f"Collecting data for 100 episodes (up to 100 steps each)...")

for ep in range(100):
    obs = eval_env.reset()
    
    # --- THE FIX: Grab the map FOR THIS SPECIFIC EPISODE ---
    current_maze_map = np.array(base_env.maze.maze_map)
    
    ep_features = []
    ep_curr_pos = []
    ep_next_pos = []
    
    for step in range(100):
        features_storage.clear()
        
        # 1. Record CURRENT Position
        curr_grid = get_agent_grid_pos(base_env)
        ep_curr_pos.append(curr_grid)
        
        # 2. Forward pass & Step
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        
        # 3. Record NEXT Position
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
    
    # --- THE FIX: Tile the map to match the number of steps in this episode ---
    all_maze_maps.append(np.tile(current_maze_map, (len(ep_features), 1, 1)))
    
    print(f"Finished Episode {ep+1}, steps: {len(ep_features)}")

# Cleanup
handle.remove()

# --- 4. Formatting Data ---
# Since episodes might end early (dones[0] == True), we use np.concatenate 
# to flatten everything into a single timeline instead of a strict 3D array.
final_features = np.concatenate(all_episode_features, axis=0) # (Total_Steps, 64, 8, 8)
final_curr_pos = np.concatenate(all_current_pos, axis=0)      # (Total_Steps, 2)
final_next_pos = np.concatenate(all_next_pos, axis=0)         # (Total_Steps, 2)
final_maze_maps = np.concatenate(all_maze_maps, axis=0)       # (Total_Steps, 8, 8)

print("\n" + "="*30)
print(f"Collection Complete!")
print(f"Features Shape:    {final_features.shape}")
print(f"Maze Map Shape:    {final_maze_maps.shape}")
print(f"Current Pos Shape: {final_curr_pos.shape}") 
print(f"Next Pos Shape:    {final_next_pos.shape}")
print("="*30)

# --- 5. Saving Data ---
save_path = "maze_probing_data_labeled.npz"
np.savez(save_path, 
         features=final_features,   
         maze_map=final_maze_maps,      # Now properly synced with the features
         agent_pos_curr=final_curr_pos, 
         agent_pos_next=final_next_pos  
         )

print(f"Data saved to {save_path}")