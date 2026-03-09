import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Data
data = np.load("maze_probing_data_labeled.npz")
features = data['features']   # (Episodes, Steps, 64, 8, 8)
maze_map = data['maze_map']   # (8, 8)

# Flatten the time dimensions for easier processing
# (Total_Steps, 64, 8, 8)
X_batch = features.reshape(-1, 64, 8, 8)
# We only need a subset to test alignment quickly (first 500 steps)
X_subset = X_batch[:500] 
# Tile the ground truth to match the subset
Y_subset = np.tile(maze_map, (500, 1, 1))

def test_alignment(transform_name, X_transformed, Y_true):
    """
    Trains a quick linear probe on the transformed data and returns accuracy.
    """
    # Flatten: (Samples * 8 * 8, 64)
    X_flat = X_transformed.transpose(0, 2, 3, 1).reshape(-1, 64)
    # Flatten: (Samples * 8 * 8)
    Y_flat = Y_true.reshape(-1)
    
    # Quick Linear Probe
    clf = LogisticRegression(max_iter=100, solver='sag') # Fast solver
    clf.fit(X_flat, Y_flat)
    
    acc = clf.score(X_flat, Y_flat)
    return acc

print(f"Baseline (Majority Class): {np.mean(Y_subset == 1):.2%}")
print("-" * 40)

# --- Define Transforms to Test ---
# We treat the feature map (8x8) as the image we are rotating/flipping
results = []

# List of (k, flip_bool)
# k: number of 90-degree rotations
# flip: whether to flip upside down
transforms = [
    (0, False), (1, False), (2, False), (3, False), # Rotations
    (0, True),  (1, True),  (2, True),  (3, True)   # Rotations + Flip
]

for k, flip in transforms:
    # Copy original features
    X_aug = X_subset.copy()
    
    # 1. Apply Flip if needed
    if flip:
        # Flip along height dimension (axis 2)
        X_aug = np.flip(X_aug, axis=2)
        name = f"Flip + Rot {k*90}"
    else:
        name = f"Original + Rot {k*90}"
        
    # 2. Apply Rotation
    # Rotate axes (2, 3) which are (H, W)
    X_aug = np.rot90(X_aug, k=k, axes=(2, 3))
    
    # 3. Test
    score = test_alignment(name, X_aug, Y_subset)
    results.append((score, name))
    print(f"{name:<20} : {score:.2%} accuracy")

# --- Check for Translation (Shift) ---
# Sometimes features are shifted by 1 pixel due to padding
print("-" * 40)
print("Checking Translations (Shifts) on Original...")

shifts = [-1, 0, 1]
for dy in shifts:
    for dx in shifts:
        if dy == 0 and dx == 0: continue
        
        # Shift the features
        X_aug = np.roll(X_subset, shift=(dy, dx), axis=(2, 3))
        
        # Mask edges (since roll wraps around, edges are invalid)
        # We just test it raw; if alignment is better, accuracy will spike regardless
        name = f"Shift y={dy}, x={dx}"
        score = test_alignment(name, X_aug, Y_subset)
        print(f"{name:<20} : {score:.2%} accuracy")

best_acc, best_name = max(results, key=lambda x: x[0])
print("\n" + "="*40)
print(f"BEST ALIGNMENT: {best_name} (Acc: {best_acc:.2%})")
print("="*40)