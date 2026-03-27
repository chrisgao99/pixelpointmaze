import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# --- Import custom components from your training file ---
# Ensure your training file is named 'train.py' or update the import accordingly
try:
    from train import DynamicAlignedPixelWrapper, CustomCombinedExtractor,DUMMY_FULL_MAP
except ImportError:
    print("Error: Could not find train.py. Ensure the training script is in the same directory.")
    exit()

# ==========================================
# 1. Configuration
# ==========================================
env_id = "PointMaze_Medium-v3"
save_name = "sac_pixelpm_aligned_64_dynamic"  # Update this to match your model's name
model_path = f"./logs/{save_name}/best_model.zip" 
video_folder = f"./logs/{save_name}/test_videos"
num_test_episodes = 10

# ==========================================
# 2. Evaluation Environment Setup
# ==========================================
def make_test_env():
    # Base Env
    env = gym.make(env_id, render_mode="rgb_array", maze_map=DUMMY_FULL_MAP)
    
    # 64x64 Wrapper (Imported from train.py)
    env = DynamicAlignedPixelWrapper(env)
    
    # Record Video
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=lambda x: True, # Record every episode
        name_prefix="test-eval"
    )
    return env

if __name__ == "__main__":
    print(f"Loading model from: {model_path}")

    eval_env = DummyVecEnv([make_test_env])
    
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='last')
    eval_env = VecTransposeImage(eval_env)

    model = SAC.load(model_path, env=eval_env)
    print(model.policy)
    # breakpoint()  

    print("\n" + "="*30)
    print("CNN LAYER FEATURE SHAPES")
    print("="*30)

    # 1. Access the observation extractor (the CNN)
    # model.policy.features_extractor is our CustomCombinedExtractor
    cnn_part = model.policy.actor.features_extractor.extractors['observation']
    
    # 2. Prepare a dummy input matching the stacked observation:
    # Shape: (Batch, Stack*Channels, Height, Width) -> (1, 12, 64, 64)
    n_stack = 4
    n_channels = 3
    dummy_input = torch.zeros(1, n_stack * n_channels, 64, 64).to(model.device)

    # 3. Pass input through layers one by one and print shapes
    current_features = dummy_input
    print(f"Input Shape:          {current_features.shape}")

    for i, layer in enumerate(cnn_part):
        current_features = layer(current_features)
        layer_name = layer.__class__.__name__
        print(f"Layer {i} ({layer_name: <10}): {current_features.shape}")

    print("="*30 + "\n")
    # breakpoint()  # Pause here to inspect the model and shapes before evaluation

    # ==========================================
    # 3. Evaluation Loop
    # ==========================================
    successes = []
    rewards = []

    # frames = []

    obs = eval_env.reset()
    # ==========================================
    # --- DEBUG BLOCK: PRINT THE MAP AND PHYSICS ---
    # ==========================================
    # 1. Reach through VecEnv and your Wrapper to the base env
    base_env = eval_env.envs[0].unwrapped
    
    # 2. Print the Logical Map (the 2D array)
    # Note: Depending on the gymnasium-robotics version, this might be ._maze_map or .maze_map
    current_map = getattr(base_env.maze, 'maze_map', None) 
    if current_map is None:
        current_map = base_env.maze._maze_map
        
    print("\n" + "="*30)
    print("LOGICAL MAZE MAP (2D Array)")
    print("="*30)
    for row in current_map:
        print(" ".join([str(int(cell)) for cell in row]))

    print(f"Starting {num_test_episodes} evaluation episodes...")

    for ep in range(num_test_episodes):
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            
            episode_reward += reward[0]
            done = dones[0]
            
            if done:
                is_success = infos[0].get("success", False)
                successes.append(is_success)
                rewards.append(episode_reward)
                
                status = "SUCCESS" if is_success else "FAIL"
                print(f"Episode {ep + 1}: Reward: {episode_reward:.2f} | Status: {status}")

                # ==========================================
                # --- DEBUG BLOCK: PRINT THE NEW MAP ---
                # ==========================================
                # Because the VecEnv just auto-reset, base_env now holds the NEW map!
                base_env = eval_env.envs[0].unwrapped
                
                current_map = getattr(base_env.maze, 'maze_map', None) 
                if current_map is None:
                    current_map = base_env.maze._maze_map
                    
                print(f"\nLOGICAL MAZE MAP (Episode {ep + 2})") # Next episode's map
                for row in current_map:
                    print(" ".join([str(int(cell)) for cell in row]))
                print("-" * 30)
                


    # ==========================================
    # 4. Final Report
    # ==========================================
    eval_env.close()
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Success Rate:   {np.mean(successes) * 100:.1f}%")
    
    if False in successes:
        print("Failed Episodes indices:", [i+1 for i, s in enumerate(successes) if not s])
    print("="*30)