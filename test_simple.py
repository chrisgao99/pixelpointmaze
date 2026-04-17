import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Import custom components from your training file ---
try:
    from train_simple import DynamicWallMatrixWrapper, CustomCombinedExtractor, DUMMY_FULL_MAP
except ImportError:
    print("Error: Could not find train.py. Ensure the training script is in the same directory.")
    exit()

# ==========================================
# 1. Configuration
# ==========================================
env_id = "PointMaze_Medium-v3"
# 确保这里的文件名与你训练保存的模型名字一致
save_name = "sac_matrixpm_forced_bypass_touch_penalty04"
model_path = f"./logs/{save_name}/best_model.zip" 
video_folder = f"./logs/{save_name}/test_videos"
num_test_episodes = 10

# ==========================================
# 2. Evaluation Environment Setup
# ==========================================
def make_test_env():
    # Base Env
    env = gym.make(env_id, render_mode="rgb_array", maze_map=DUMMY_FULL_MAP)
    
    # 替换为矩阵的 Wrapper
    env = DynamicWallMatrixWrapper(env)
    
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

    # 这里去掉了 VecFrameStack 和 VecTransposeImage
    eval_env = DummyVecEnv([make_test_env])

    model = SAC.load(model_path, env=eval_env)
    print("\nPolicy Architecture:")
    print(model.policy)
    print("="*30 + "\n")

    # ==========================================
    # 3. Evaluation Loop
    # ==========================================
    successes = []
    rewards = []

    obs = eval_env.reset()
    
    # ==========================================
    # --- DEBUG BLOCK: PRINT THE MAP ---
    # ==========================================
    base_env = eval_env.envs[0].unwrapped
    
    current_map = getattr(base_env.maze, 'maze_map', None) 
    if current_map is None:
        current_map = base_env.maze._maze_map
        
    print("\n" + "="*30)
    print("LOGICAL MAZE MAP (2D Array)")
    print("="*30)
    for row in current_map:
        print(" ".join([str(int(cell)) for cell in row]))

    print(f"Starting {num_test_episodes} evaluation episodes...\n")

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
                base_env = eval_env.envs[0].unwrapped
                
                current_map = getattr(base_env.maze, 'maze_map', None) 
                if current_map is None:
                    current_map = base_env.maze._maze_map
                    
                print(f"\nLOGICAL MAZE MAP (Episode {ep + 2})") 
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