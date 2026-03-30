import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
data = np.load("maze_probing_data_labeled.npz")
features = data['features']   # Shape: (Episodes, Steps, 64, 8, 8)
maze_map = data['maze_map']   # Shape: (8, 8)

# 2. Create the "Feature Ghost"
# We want to see which pixels are active when walls are present.
# Since we don't know which channel is "wall", we take the variance or max activation across time.
# High variance usually means "something interesting happens here" (like a wall appearing/disappearing or agent moving).
feature_variance = np.var(features, axis=(0, 1, 2)) # Variance across Eps, Steps, and Channels

# Normalize for plotting (0 to 1)
feature_img = (feature_variance - feature_variance.min()) / (feature_variance.max() - feature_variance.min())

# 3. Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot A: What the CNN Actually Sees (The Ghost)
sns.heatmap(feature_img, cmap="viridis", ax=axes[0], square=True, cbar=False)
axes[0].set_title("CNN Feature Activity (Variance)\n(What the network sees)", fontsize=14)
axes[0].set_xlabel("CNN X-Axis")
axes[0].set_ylabel("CNN Y-Axis")

# Plot B: The Ground Truth Maze
# We try to align orientation visually
sns.heatmap(maze_map, cmap="Greys", ax=axes[1], square=True, cbar=False, linewidths=1, linecolor='black')
axes[1].set_title("Ground Truth Maze Map\n(0=Path, 1=Wall)", fontsize=14)
axes[1].set_xlabel("Maze Column")
axes[1].set_ylabel("Maze Row")

plt.suptitle("Alignment Check: Does the Activity Match the Map?", fontsize=16)
plt.tight_layout()
plt.savefig("maze_feature_alignment.png")
print("\nVisualization saved as 'maze_feature_alignment.png'")