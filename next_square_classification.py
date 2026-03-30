import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data...")
data = np.load("maze_probing_data_labeled.npz")
features = data['features']             # (Episodes, Steps, 64, 8, 8)
agent_pos_curr = data['agent_pos_curr'] # (Episodes, Steps, 2)
agent_pos_next = data['agent_pos_next'] # (Episodes, Steps, 2)

N_episodes, N_steps = features.shape[:2]
N_samples = N_episodes * N_steps

# Flatten the episodic dimension to iterate step-by-step
features_flat = features.reshape(N_samples, 64, 8, 8)
curr_flat = agent_pos_curr.reshape(-1, 2)
next_flat = agent_pos_next.reshape(-1, 2)

X_binary = []
y_binary = []

# ==========================================
# 2. Extract Cell-Specific Samples
# ==========================================
print("Extracting positive samples and neighboring negative samples...")

for i in range(N_samples):
    r_c, c_c = curr_flat[i]
    r_n, c_n = next_flat[i]
    
    # Safety clip to ensure indices are within [0, 7]
    r_c, c_c = np.clip([r_c, c_c], 0, 7)
    r_n, c_n = np.clip([r_n, c_n], 0, 7)
    
    # --- POSITIVE SAMPLE ---
    # The 64-d feature vector at the exact next step location
    X_binary.append(features_flat[i, :, r_n, c_n])
    y_binary.append(1)
    
    # --- NEGATIVE SAMPLES ---
    # Look at the 3x3 grid around the CURRENT position
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r = r_c + dr
            c = c_c + dc
            
            # Check if the neighbor is within the 8x8 grid bounds
            if 0 <= r < 8 and 0 <= c < 8:
                # Exclude the actual next step (so we don't label a positive as a negative)
                if r == r_n and c == c_n:
                    continue
                    
                X_binary.append(features_flat[i, :, r, c])
                y_binary.append(0)

X_binary = np.array(X_binary)
y_binary = np.array(y_binary)

print(f"Total samples created: {len(y_binary)}")
print(f"Positives (Class 1): {np.sum(y_binary == 1)}")
print(f"Negatives (Class 0): {np.sum(y_binary == 0)}")

# ==========================================
# 3. Train / Test Split & Probing
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

print(f"\nTraining localized binary probe on {len(X_train)} samples...")
# Using class_weight='balanced' because we have ~8x more negatives than positives
probe = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
probe.fit(X_train, y_train)

# ==========================================
# 4. Evaluation
# ==========================================
y_pred = probe.predict(X_test)

print("\n" + "="*50)
print(f"Binary Probing Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("="*50)

# Accuracy can be misleading with imbalanced data, so we print the full report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Next Step (0)", "Next Step (1)"]))

# ==========================================
# 5. Save Raw Weights (NumPy)
# ==========================================
print("\nSaving raw weights...")
weights_filename = "single_cell_next_step_probe_weights.npz"

# probe.coef_ holds the weights (shape: 1 x n_features)
# probe.intercept_ holds the bias term (shape: 1,)
# probe.classes_ keeps track of the class labels (usually [0, 1])
np.savez(weights_filename, 
         weights=probe.coef_, 
         bias=probe.intercept_, 
         classes=probe.classes_)

print(f"Raw weights saved successfully to '{weights_filename}'.")