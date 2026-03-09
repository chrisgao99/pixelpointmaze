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
    from train import DynamicAlignedPixelWrapper, CustomCombinedExtractor,DUMMY_9_WALL_MAP
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
    env = gym.make(env_id, render_mode="rgb_array", maze_map=DUMMY_9_WALL_MAP)
    
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

    # 1. Create the vectorized environment
    eval_env = DummyVecEnv([make_test_env])
    
    # 2. Stack and Transpose (Must match training exactly: 4 frames, CHW)
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='last')
    eval_env = VecTransposeImage(eval_env)

    # 3. Load Model
    # Note: policy_kwargs are stored inside the zip, so SB3 handles the CustomCombinedExtractor
    model = SAC.load(model_path, env=eval_env)
    print(model.policy)
    breakpoint()  # Pause here to inspect the loaded model and its policy architecture

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
    print(obs["observation"].shape)  # Should be (1, 12, 64, 64)
    # breakpoint()
    # frames.append(obs["observation"][0])  # Append the first observation (remove batch dimension)
    print(f"Starting {num_test_episodes} evaluation episodes...")

    for ep in range(num_test_episodes):
        done = False
        episode_reward = 0
        
        # Note: VecEnv resets automatically, so we track episodes manually
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            # frames.append(obs["observation"][0])  # Append the first observation of the new step
            
            episode_reward += reward[0]
            done = dones[0]
            
            if done:
                # In SB3 VecEnv, the success info is usually in the 'info' dict
                # of the last step before the auto-reset
                is_success = infos[0].get("success", False)
                successes.append(is_success)
                rewards.append(episode_reward)
                
                status = "SUCCESS" if is_success else "FAIL"
                print(f"Episode {ep + 1}: Reward: {episode_reward:.2f} | Status: {status}")
                # now save an extra video from the frames collected
                # try:
                #     import imageio
                # except ImportError:
                #     print("Error: Please install imageio (pip install imageio) to save videos.")
                #     break

                # video_filename = os.path.join(video_folder, f"eval_ep_{ep+1}_{status}.mp4")
                
                # # 1. Handle VecEnv Auto-Reset:
                # # The last frame in 'frames' is actually the start of the NEXT episode.
                # # We remove it for the video, but keep it to start the list for the next loop.
                # next_ep_start = frames.pop()
                
                # processed_frames = []
                # for f in frames:
                #     # f shape is (12, 64, 64) -> (Channels * Stack, Height, Width)
                #     # We want the MOST RECENT frame (the last 3 channels in the stack)
                #     # Shape becomes (3, 64, 64)
                #     current_frame_chw = f[-3:, :, :]

                #     # 2. Transpose to HWC for video saving: (64, 64, 3)
                #     current_frame_hwc = np.transpose(current_frame_chw, (1, 2, 0))

                #     # 3. Denormalize/Cast if necessary
                #     # If your wrapper outputs floats (0.0 to 1.0), scale to 255
                #     if current_frame_hwc.max() <= 1.0 and current_frame_hwc.dtype != np.uint8:
                #         current_frame_hwc = (current_frame_hwc * 255).astype(np.uint8)
                #     else:
                #         current_frame_hwc = current_frame_hwc.astype(np.uint8)

                #     processed_frames.append(current_frame_hwc)

                # # 4. Save Video
                # # macro_block_size=None prevents FFmpeg errors on non-standard resolutions
                # imageio.mimsave(video_filename, processed_frames, fps=30, macro_block_size=None)
                # print(f"  > Video saved to: {video_filename}")

                # # 5. Reset frames list for the next episode using the observation we popped
                # frames = [next_ep_start]

                


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