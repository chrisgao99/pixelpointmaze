import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import gymnasium as gym
import mujoco 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# --- Import custom components ---
try:
    from train import AlignedPixelWrapper
except ImportError:
    print("Warning: train.py not found. Ensure AlignedPixelWrapper is available.")

# ==========================================
# 1. Configuration
# ==========================================
env_id = "PointMaze_Medium-v3"
save_name = "sac_pixelpm_aligned_64"
model_path = f"./logs/{save_name}/best_model.zip"
probe_path = "maze_probe_linear.pkl"
probe_path = "agentpos_probe_linear.pkl"  # Alternative probe trained on agent's features

# --- TOGGLE MEMORY HERE ---
MEMORY_MODE = False  
# True:  Agent remembers all past cells + human inputs forever.
# False: Agent remembers nothing. Human inputs stay active only until visited.

intervention_strength = 0.01  

START_POS = np.array([2.1, 2.3]) 
GOAL_POS = np.array([-2.8, 2.0]) 
hack = True  # If True, forcibly set start/goal positions in the environment for consistency.

# Load Probe
clf_linear = joblib.load(probe_path)
W = torch.from_numpy(clf_linear.coef_.flatten()).float()
b = float(clf_linear.intercept_.item()) 

# ==========================================
# 2. Logic: Tracker & Hook
# ==========================================
class TrajectoryTracker:
    def __init__(self, grid_size=8, memory=True):
        self.grid_size = grid_size
        self.memory = memory
        
        # Mask 1: Accumulates everything (used if memory=True)
        self.history_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Mask 2: Transient human targets (used if memory=False)
        self.target_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        self.current_r = 0
        self.current_c = 0
        
    def reset(self):
        self.history_mask.fill(False)
        self.target_mask.fill(False)
        
    def _coords_to_grid(self, x, y):
        c = int(np.floor(x - (-4)))
        r = int(np.floor(4 - y))
        return r, c

    def update_position(self, pos_x, pos_y):
        r, c = self._coords_to_grid(pos_x, pos_y)
        self.current_r = r
        self.current_c = c
        
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return

        if self.memory:
            # Memory ON: Permanently mark visited cells
            self.history_mask[r, c] = True
        else:
            # Memory OFF: Check if we reached a human-set target
            if self.target_mask[r, c]:
                print(f"  -> Agent reached target at ({r},{c}). Intervention cleared.")
                self.target_mask[r, c] = False

    def manual_intervention(self, direction):
        r, c = self.current_r, self.current_c
        if direction == 'up': r -= 1
        elif direction == 'down': r += 1
        elif direction == 'left': c -= 1
        elif direction == 'right': c += 1
            
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            print(f"Human Intervention at ({r}, {c})")
            
            if self.memory:
                self.history_mask[r, c] = True
            else:
                self.target_mask[r, c] = True

    def get_active_mask(self):
        if self.memory:
            return self.history_mask
        else:
            return self.target_mask

class InterventionHook:
    def __init__(self, tracker, W, strength=0.01):
        self.tracker = tracker
        self.W = W
        self.strength = strength
        self.latest_features = None

    def __call__(self, module, input, output):
        device = output.device
        numpy_mask = self.tracker.get_active_mask()
        mask_tensor = torch.from_numpy(numpy_mask).to(device).unsqueeze(0).unsqueeze(0).float()
        
        w_vec = self.W.to(device).view(1, 64, 1, 1)
        
        # Subtract Wall Vector where mask is True
        # (Making the agent think it is NOT a wall)
        delta = mask_tensor * (-1.0 * self.strength * w_vec)
        
        modified_output = output + delta
        self.latest_features = modified_output.detach().cpu()
        return modified_output

# ==========================================
# 3. Interactive Visualization
# ==========================================
class LiveVisualizer:
    def __init__(self, tracker):
        self.tracker = tracker
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        self.im1 = self.ax1.imshow(np.zeros((64, 64, 3)))
        self.ax1.set_title("Agent View")
        self.ax1.axis('off')
        
        # RESTORED: RdBu_r colormap (Red=Wall, Blue=Path)
        self.im2 = self.ax2.imshow(np.zeros((8, 8)), cmap='RdBu_r', vmin=0, vmax=1)
        self.ax2.set_title("Internal Representation\n(Red=Wall, Blue=Path)")
        self.ax2.axis('off')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print("\nControls activated: Click on the plot window and use Arrow Keys!")

    def on_key(self, event):
        if event.key in ['up', 'down', 'left', 'right']:
            self.tracker.manual_intervention(event.key)

    def update(self, env_img, prob_map, step):
        self.im1.set_data(env_img)
        
        # Show the probability map (The Red/Blue view)
        self.im2.set_data(prob_map)
        
        status = "ON" if self.tracker.memory else "OFF"
        self.ax1.set_title(f"Step: {step} | Memory: {status}")
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==========================================
# 4. Helper
# ==========================================
def get_prob_map(features_tensor, W, b):
    if features_tensor is None: return np.zeros((8, 8))
    feat = features_tensor[0]
    feat_flat = feat.view(64, -1).T 
    logits = torch.matmul(feat_flat, W) + b
    return torch.sigmoid(logits).view(8, 8).numpy()

# ==========================================
# 5. Main Loop
# ==========================================
if __name__ == "__main__":
    def make_env():
        env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=500)
        env = AlignedPixelWrapper(env)
        return env

    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='last')
    eval_env = VecTransposeImage(eval_env)
    
    model = SAC.load(model_path, env=eval_env)
    
    tracker = TrajectoryTracker(memory=MEMORY_MODE)
    hook = InterventionHook(tracker, W, intervention_strength)
    visualizer = LiveVisualizer(tracker)
    
    cnn_block = model.policy.actor.features_extractor.extractors['observation']
    handle = cnn_block[3].register_forward_hook(hook)
    
    num_episodes = 5
    for episode in range(num_episodes):
        obs = eval_env.reset()
        
        # --- HACK: Force Start/Goal positions ---
        if hack:
            raw_env = eval_env.envs[0].unwrapped
            qpos = raw_env.data.qpos.copy()
            qpos[:2] = START_POS
            raw_env.data.qpos[:] = qpos
            raw_env.data.qvel[:] = 0
            raw_env.goal = GOAL_POS
        
            try:
                target_site_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_SITE, "target")
                if target_site_id != -1:
                    new_site_pos = np.array([GOAL_POS[0], GOAL_POS[1], 0.0])
                    raw_env.model.site_pos[target_site_id] = new_site_pos
                    raw_env.data.site_xpos[target_site_id] = new_site_pos
            except: pass

            mujoco.mj_forward(raw_env.model, raw_env.data)
            zero_action = np.zeros(raw_env.action_space.shape)
            obs, _, _, _ = eval_env.step([zero_action]) 
            # ---------------------------------------

        tracker.reset()
        done = False
        step = 0
        
        print(f"=== Episode {episode + 1} (Memory={'ON' if MEMORY_MODE else 'OFF'}) ===")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            done = dones[0]
            step += 1
            
            current_pos = obs['achieved_goal'][0]
            if step > 2:
                tracker.update_position(current_pos[0], current_pos[1])
            
            # 1. Get Image
            env_img = eval_env.envs[0].render()
            
            # 2. Get Prob Map (Red/Blue) using latest hooked features
            prob_map = get_prob_map(hook.latest_features, W, b)
            
            # 3. Update Visualizer
            visualizer.update(env_img, prob_map, step)
            plt.pause(0.05) 

    handle.remove()
    eval_env.close()