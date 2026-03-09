# pixelpointmaze

This repo provides code to train a pixel-based SAC agent on PointMaze, probe its CNN features to check if they encode wall positions, and run an interactive demo where a human can steer the agent at runtime by manipulating its internal features with arrow keys.

---

## Pipeline Overview

### 1. `train.py` — Train the agent

Trains a SAC agent with Hindsight Experience Replay (HER) on `PointMaze_Medium-v3` using pixel observations instead of state vectors.

Key components:
- **`generate_single_random_maze()`** — Procedurally generates connected 8×8 mazes with exactly 9 inner walls.
- **`DUMMY_9_WALL_MAP`** — A fixed template maze required so MuJoCo compiles the right number of movable wall blocks at startup.
- **`DynamicAlignedPixelWrapper`** — A `gym.ObservationWrapper` that replaces the default state observation with a top-down 64×64 RGB render. On each episode reset, it re-randomizes the maze layout by physically repositioning MuJoCo wall geoms.
- **`CustomCombinedExtractor`** — A CNN-based feature extractor for dict observations: processes the 64×64 image through two deliberately sized conv layers (4×4 stride-4, then 2×2 stride-2) that reduce the spatial dimensions to exactly 8×8 with 64 channels. This is intentional — the 8×8 feature map has a one-to-one spatial correspondence with the 8×8 maze grid, so each CNN cell encodes exactly one maze block. Goal vectors are flattened separately.
- The main script trains with 10 parallel environments, 4-frame stacking, for 3M timesteps. Checkpoints and the best model are saved under `logs/`.

### 2. `test.py` — Evaluate and inspect the trained model

Loads a saved SAC model and runs it for several episodes in the pixel PointMaze environment.

- Records video of every test episode using `RecordVideo`.
- Includes a diagnostic section that passes a dummy tensor through the CNN layer-by-layer and prints intermediate feature shapes, useful for verifying the architecture before probing.

### 3. `collect_align.py` — Collect CNN features with position labels

Runs the trained agent for 100 episodes (100 steps each) and collects data for probing the internal representations.

- Registers a forward hook on a specific CNN layer to capture the 64×8×8 intermediate feature maps at every step.
- Records the agent's grid position (row, col) before and after each step using the maze's coordinate converter.
- Saves everything to `maze_probing_data_labeled.npz`: features, maze wall map, and agent position labels.

### 4. `feat_wall_*.py` — Analyze feature-wall alignment

A set of diagnostic scripts that probe whether the CNN's 8×8 spatial feature maps encode wall positions.

- **`feat_wall_diagnosis.py`** — Computes per-pixel feature variance across all timesteps and plots it as a heatmap alongside the ground truth maze map, for a quick visual alignment check.
- **`feat_wall_diagnosis2.py`** — Trains a logistic regression classifier on all spatial feature vectors to predict wall vs. path, then renders the predicted wall-probability map (8×8) as `maze_wall_diagnosis.png`.
- **`feat_wall_align.py`** — Systematically tests all 8 combinations of rotations (0°/90°/180°/270°) and vertical flips on the feature map and runs a linear probe for each, identifying the geometric transform that best aligns the feature grid to the maze map. Also checks integer pixel shifts.
- **`feat_wall_intv_all.py`** — Loads the trained linear probe and applies a global feature-space intervention: nudges CNN activations along the learned wall/path direction for all wall and path samples respectively, then re-evaluates accuracy and plots error maps before and after intervention.

### 5. `human-rl.py` — Real-time human intervention

An interactive session where a human can steer the agent at runtime by manipulating its internal CNN features.

- Loads the trained SAC model and linear wall probe.
- Opens a matplotlib window showing (1) the agent's live camera view and (2) an 8×8 wall-probability map derived from the live CNN features.
- **Arrow key input**: pressing an arrow key marks the adjacent grid cell as a "virtual wall" by subtracting the probe's learned wall direction vector from the CNN's feature map at that location via a forward hook. This causes the agent to perceive and avoid the cell as if it were a wall.
- **`TrajectoryTracker`** manages which cells are currently intervened on, with two modes:
  - `MEMORY_MODE=True`: Interventions are permanent for the episode.
  - `MEMORY_MODE=False`: An intervention clears itself once the agent physically reaches that cell.
- Start and goal positions can be fixed via the `hack` flag for reproducible demos.
