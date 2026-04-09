import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# 1. Load Data
data = np.load("maze_probing_data_labeled.npz")
features = data['features']   # Shape is now: (Total_Steps, 64, 8, 8)
maze_maps = data['maze_map']  # Shape is now: (Total_Steps, 8, 8)

n_total_steps, n_channels, height, width = features.shape

print(f"Loaded {n_total_steps} total steps.")

# --- NEW STRATEGY: Feature Similarity ---
# 1. Train Global "Is this a Wall?" Classifier
# Flatten spatial dims: (Total_Steps * 8 * 8, 64)
# We transpose to (Total_Steps, 8, 8, 64) before reshaping so pixels align correctly
X_flat = features.transpose(0, 2, 3, 1).reshape(-1, 64) 
Y_flat = maze_maps.reshape(-1)

print("Training Global Wall Detector Probe...")
clf = LogisticRegression(max_iter=500, solver='sag')

# Use 10% of data to train quickly
indices = np.random.choice(len(Y_flat), size=len(Y_flat)//10, replace=False)
clf.fit(X_flat[indices], Y_flat[indices])

score = clf.score(X_flat, Y_flat)
print(f"Global Probe Accuracy: {score:.4f}")

# 2. Apply this detector to the Spatial Grid
# Let's take the average features over ALL time to see the "average" wall perception
# Note: Since the map is dynamic now, averaging over time might blur things if walls move.
# Instead, let's visualize the LAST step's features against the LAST step's map.
example_features = features[-1] # (64, 8, 8)
example_map = maze_maps[-1]     # (8, 8)

# Reshape for prediction: (64_pixels, 64_channels)
flat_spatial_features = example_features.transpose(1, 2, 0).reshape(-1, 64)

# Predict "Wall Probability" for each of the 64 pixels
wall_probs = clf.predict_proba(flat_spatial_features)[:, 1]
wall_probs_map = wall_probs.reshape(8, 8)

# 3. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Ground Truth
sns.heatmap(example_map, annot=True, cmap="Greys", cbar=False, ax=axes[0], square=True,
            linewidths=1, linecolor='black')
axes[0].set_title("Ground Truth Maze (Last Step)")

# CNN Prediction
sns.heatmap(wall_probs_map, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], square=True,
            linewidths=1, linecolor='gray')
axes[1].set_title("Where the CNN 'Thinks' Walls Are\n(1.0 = High Confidence Wall)")

plt.tight_layout()
plt.savefig("maze_wall_diagnosis.png")
print("Saved diagnosis plot to maze_wall_diagnosis.png")