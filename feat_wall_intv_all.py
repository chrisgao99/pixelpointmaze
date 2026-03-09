import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data and Model ---
data_path = "maze_probing_data.npz"
model_path = "maze_probe_linear.pkl"

data = np.load(data_path)
features = data['features']
maze_map = data['maze_map']

clf_linear = joblib.load(model_path)

# Prepare Data
n_eps, n_steps, n_channels, height, width = features.shape
X = features.transpose(0, 1, 3, 4, 2).reshape(-1, n_channels)
Y_true = np.tile(maze_map, (n_eps * n_steps, 1, 1)).reshape(-1)

# --- 2. Original Prediction ---
print("Running original inference...")
Y_pred_orig = clf_linear.predict(X)

# Identify which samples were ALREADY CORRECT
correct_indices = np.where(Y_pred_orig == Y_true)[0]
original_acc = len(correct_indices) / len(Y_true) * 100
print(f"Original Global Accuracy: {original_acc:.2f}%")
print(f"Number of originally correct samples: {len(correct_indices)}")

# --- 3. Perform Global Intervention ---
# We apply the push to ALL samples, not just the wrong ones.
print("\nApplying intervention to ALL samples...")

W = clf_linear.coef_.flatten()
intervention_strength = 0.001

X_all_intervened = X.copy()

# 1. Push ALL Ground Truth Walls towards Wall (+W)
#    This includes Walls we got right, and Walls we got wrong.
mask_walls = (Y_true == 1)
X_all_intervened[mask_walls] += (W * intervention_strength)

# 2. Push ALL Ground Truth Paths towards Path (-W)
#    This includes Paths we got right, and Paths we got wrong.
mask_paths = (Y_true == 0)
X_all_intervened[mask_paths] -= (W * intervention_strength)

# --- 4. Re-evaluate ---
print("Running inference on globally intervened features...")
Y_pred_all_new = clf_linear.predict(X_all_intervened)

# --- 5. Critical Check: Did we break the correct ones? ---
# We check the accuracy ONLY on the subset of data that was originally correct.
# If this drops below 100%, we broke something.
Y_pred_subset = Y_pred_all_new[correct_indices]
Y_true_subset = Y_true[correct_indices]

subset_acc = np.mean(Y_pred_subset == Y_true_subset) * 100
print(f"\n--- SAFETY CHECK ---")
print(f"Accuracy on originally CORRECT samples after intervention: {subset_acc:.2f}%")

if subset_acc == 100.0:
    print("SUCCESS: Pushing correct samples further in the correct direction did not break them.")
else:
    print("WARNING: Some correct samples were flipped!")

# Global Accuracy after intervention
global_acc_new = np.mean(Y_pred_all_new == Y_true) * 100
print(f"New Global Accuracy (All Samples): {global_acc_new:.2f}%")

# --- 6. Helper for Visualization ---
def get_error_map(y_pred, y_true, height, width):
    y_pred_spatial = y_pred.reshape(-1, height, width)
    y_true_spatial = y_true.reshape(-1, height, width)
    mask = (y_pred_spatial != y_true_spatial)
    return np.mean(mask, axis=0) * 100

error_map_orig = get_error_map(Y_pred_orig, Y_true, height, width)
error_map_new = get_error_map(Y_pred_all_new, Y_true, height, width)

# --- 7. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Ground Truth
sns.heatmap(maze_map, annot=True, fmt="d", cmap="Greys", cbar=False, 
            ax=axes[0], square=True, linewidths=.5, linecolor='black')
axes[0].set_title("Ground Truth Maze", fontsize=14)

# Plot 2: Original Errors
sns.heatmap(error_map_orig, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100,
            ax=axes[1], square=True, linewidths=.5, linecolor='gray')
axes[1].set_title(f"Original Errors (%)\nAcc: {original_acc:.1f}%", fontsize=14)

# Plot 3: Global Intervention Errors
sns.heatmap(error_map_new, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100,
            ax=axes[2], square=True, linewidths=.5, linecolor='gray')
axes[2].set_title(f"Errors After Global Intervention (%)\nAcc: {global_acc_new:.1f}%", fontsize=14)

plt.tight_layout()
plt.savefig(f"maze_global_intervention_check_st{intervention_strength}.png")
print(f"\nVisualization saved as 'maze_global_intervention_check_st{intervention_strength}.png'")
# plt.show()