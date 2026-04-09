import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
data = np.load("maze_probing_data_labeled.npz")
features = data['features']   # Shape: (Total_Steps, 64, 8, 8)
maze_maps = data['maze_map']  # Shape: (Total_Steps, 8, 8)

# 2. Isolate a Single Episode
# Since maps change every episode, we can't take variance over the whole dataset.
# Let's isolate the steps from the very last episode.
last_map = maze_maps[-1]

# Find all consecutive steps at the end that share this exact map
is_same_map = np.all(maze_maps == last_map, axis=(1, 2))
# Get indices where the map is the same as the last step
episode_indices = np.where(is_same_map)[0] 

# Extract features just for this specific maze layout
# Shape becomes: (Steps_in_last_ep, 64, 8, 8)
ep_features = features[episode_indices]

# 3. Create the "Feature Ghost"
# Calculate variance across Time (axis 0) and Channels (axis 1)
# This leaves us with an (8, 8) spatial grid.
feature_variance = np.var(ep_features, axis=(0, 1))

# Normalize for plotting (0 to 1)
feature_img = (feature_variance - feature_variance.min()) / (feature_variance.max() - feature_variance.min())

# 4. Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot A: What the CNN Actually Sees (The Ghost)
sns.heatmap(feature_img, cmap="viridis", ax=axes[0], square=True, cbar=False)
axes[0].set_title(f"CNN Feature Activity (Variance)\n(Last Episode: {len(episode_indices)} steps)", fontsize=14)
axes[0].set_xlabel("CNN X-Axis")
axes[0].set_ylabel("CNN Y-Axis")

# Plot B: The Ground Truth Maze
# We try to align orientation visually
sns.heatmap(last_map, cmap="Greys", ax=axes[1], square=True, cbar=False, linewidths=1, linecolor='black')
axes[1].set_title("Ground Truth Maze Map\n(0=Path, 1=Wall)", fontsize=14)
axes[1].set_xlabel("Maze Column")
axes[1].set_ylabel("Maze Row")

plt.suptitle("Alignment Check: Does the Activity Match the Map?", fontsize=16)
plt.tight_layout()
plt.savefig("maze_feature_alignment.png")
print("\nVisualization saved as 'maze_feature_alignment.png'")