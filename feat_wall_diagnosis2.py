import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# 1. Load Data
data = np.load("maze_probing_data_labeled.npz")
features = data['features']   # (Episodes, Steps, 64, 8, 8)
maze_map = data['maze_map']   # (8, 8)

n_eps, n_steps, n_channels, height, width = features.shape

# Flatten time: (Total_Samples, 64, 8, 8)
X_all = features.reshape(-1, 64, height, width)
# Flatten labels: (Total_Samples, 8, 8)
Y_all = np.tile(maze_map, (n_eps * n_steps, 1, 1))

# 2. Pick specific "Probe Targets" (Distinct spots in the maze)
# We look for a wall in the top-left, middle, and bottom-right to see if the shift is consistent.
# Let's find valid wall coordinates automatically
wall_coords = np.argwhere(maze_map == 1)
path_coords = np.argwhere(maze_map == 0)

# Pick 3 representative points
targets = [
    ("Top-Left Wall", wall_coords[1]), # Skip 0 (border) to be safe
    ("Center Wall", wall_coords[len(wall_coords)//2]),
    ("Bottom-Right Wall", wall_coords[-2])
]

print(f"Checking alignment for targets: {[t[1] for t in targets]}")

# 3. Correlation Search
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, (target_y, target_x)) in enumerate(targets):
    # The Ground Truth for THIS specific cell (0 or 1 across all timesteps)
    # Note: Since the map is static, this is actually constant 1 for walls.
    # To find correlation, we need VARIANCE. 
    # A static label (always wall) cannot be correlated with changing features.
    
    # CRITICAL FIX: We cannot correlate with a static map because the label never changes.
    # We must correlate with "Agent is near this wall" or simply look for 
    # feature STABILITY.
    
    # ALTERNATIVE APPROACH: Feature Activation Heatmap
    # If the CNN sees a wall at (target_y, target_x), the features at that location 
    # should be fundamentally different from features at path locations.
    
    # Let's try a diff approach: "Where does the feature map look most like a WALL feature?"
    # 1. Take the average feature vector of ALL walls.
    # 2. Compare every pixel's feature vector to that "Wall Prototype".
    pass 

print("Switching strategy: Static maps have no variance to correlate.")
print("Calculating 'Wall-ness' score for every pixel...")

# --- NEW STRATEGY: Feature Similarity ---
# 1. Identify "Known Wall" pixels and "Known Path" pixels in the CNN grid (assuming 1:1 mapping)
# If mapping is wrong, this will be noisy, but the signal should peak where the REAL wall is.

# Let's look at the VARIANCE map again, but pixel-wise.
# Or better: Train a global classifier on ALL pixels mixed together, 
# then ask it to classify every pixel in the grid.

# 1. Train Global "Is this a Wall?" Classifier
# We assume *some* pixels are aligned. We flatten everything.
X_flat = features.transpose(0, 1, 3, 4, 2).reshape(-1, 64) # (Samples*8*8, 64)
Y_flat = np.tile(maze_map, (n_eps * n_steps, 1, 1)).reshape(-1)

# Train on a small subset to get a "General Wall Detector"
clf = LogisticRegression(max_iter=200, solver='sag')
# Use only 10% of data to train broadly
indices = np.random.choice(len(Y_flat), size=len(Y_flat)//10, replace=False)
clf.fit(X_flat[indices], Y_flat[indices])

# 2. Apply this detector to the Spatial Grid
# We take the average features for each (8,8) cell over all time
avg_features = np.mean(features, axis=(0, 1)) # (64, 8, 8)
# Reshape for prediction: (64_pixels, 64_channels)
flat_spatial_features = avg_features.reshape(64, -1).T # (64, 64)

# Predict "Wall Probability" for each of the 64 pixels
# shape: (64, 2) -> take column 1 (Probability of Wall)
wall_probs = clf.predict_proba(flat_spatial_features)[:, 1]
wall_probs_map = wall_probs.reshape(8, 8)

# 3. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Ground Truth
sns.heatmap(maze_map, annot=True, cmap="Greys", cbar=False, ax=axes[0], square=True,
            linewidths=1, linecolor='black')
axes[0].set_title("Ground Truth Maze (Target)")

# CNN Prediction
sns.heatmap(wall_probs_map, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], square=True,
            linewidths=1, linecolor='gray')
axes[1].set_title("Where the CNN 'Thinks' Walls Are\n(1.0 = High Confidence Wall)")

plt.tight_layout()
plt.savefig("maze_wall_diagnosis.png")