import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Load and Prepare the Data
# ==========================================
print("Loading data...")
data = np.load("maze_probing_data_labeled.npz")
features = data['features']             # Shape: (Episodes, Steps, 64, 8, 8)
agent_pos_next = data['agent_pos_next'] # Shape: (Episodes, Steps, 2)

# Get dataset dimensions
N_episodes, N_steps = features.shape[:2]
N_samples = N_episodes * N_steps

# Flatten the (Episodes, Steps) into a single dimension
# and flatten the 64x8x8 CNN map into a 4096-length feature vector
X = features.reshape(N_samples, -1) 
print(f"Feature matrix X shape: {X.shape}") # Should be (N_samples, 4096)

# Convert the 2D (row, col) coordinates into a 1D class ID (0 to 63)
# Formula: class_id = row * 8 + col
Y_coords = agent_pos_next.reshape(N_samples, 2)
Y = Y_coords[:, 0] * 8 + Y_coords[:, 1]
print(f"Target vector Y shape: {Y.shape}")  # Should be (N_samples,)

# ==========================================
# 2. Train / Test Split
# ==========================================
# We keep 20% of the data unseen to test if the probe actually generalizes
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==========================================
# 3. Train the Linear Probe
# ==========================================
print(f"\nTraining Linear Probe on {len(X_train)} samples...")
print("This might take a minute depending on your CPU...")

# We use Logistic Regression as our linear probe. 
# n_jobs=-1 uses all CPU cores to speed up the training of 4096 features.
probe = LogisticRegression(max_iter=1000, n_jobs=-1) 
probe.fit(X_train, y_train)

# ==========================================
# 4. Evaluate Accuracy
# ==========================================
y_pred = probe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f"Next-Step Probing Accuracy: {accuracy * 100:.2f}%")
print("="*40)

# Optional: Print a few examples to see what it's doing
print("\nSample Predictions vs Ground Truth (Class IDs 0-63):")
for i in range(5):
    pred_row, pred_col = divmod(y_pred[i], 8)
    true_row, true_col = divmod(y_test[i], 8)
    print(f"Sample {i+1} | Probe Predicted: ({pred_row}, {pred_col}) | Actual Next Step: ({true_row}, {true_col})")


# ==========================================
# 5. Save Raw Weights (NumPy)
# ==========================================
print("\nSaving raw multi-class weights...")
weights_filename = "whole_map_next_step_probe_weights.npz"

# probe.coef_ holds the weights (shape: 64 x 4096)
# probe.intercept_ holds the bias term (shape: 64,)
# probe.classes_ keeps track of the class labels ([0, 1, 2, ... 63])
np.savez(weights_filename, 
         weights=probe.coef_, 
         bias=probe.intercept_, 
         classes=probe.classes_)

print(f"Raw weights saved successfully to '{weights_filename}'.")