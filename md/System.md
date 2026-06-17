# Orbit Wars: Spatial U-Net & Matrix Head Architecture Whitepaper

This document is the comprehensive architectural whitepaper for the Orbit Wars Machine Learning bot. It details the exact mathematical transformations, tensor dimensions, neural network architecture, and data pipelines required to translate a 100x100 real-time strategy simulation into a 2D spatial grid for Deep Learning.

---

## 1. The Core Paradigm Shift

### The Problem with Seq2Seq and Edge-Scoring
In our previous iterations, we used an Autoregressive Seq2Seq Transformer. While mathematically elegant, it suffered from two fatal flaws in a Kaggle simulation:
1. **Inference Speed:** To issue 10 fleet commands in a single turn, the model had to run 10 complete forward passes. In the late-game, this easily triggers Kaggle's 3-second timeout limit, resulting in instant death.
2. **Error Cascades:** If the model predicts a bad "Target" on step 1, the following predictions for "Source" and "Fraction" are mathematically doomed to fail.

### The Spatial Matrix Solution
We reformulated Orbit Wars into a **Computer Vision & Matrix Prediction** problem. By mapping the galaxy to a 2D pixel grid, we process the entire board state using a blazing-fast Convolutional U-Net. Then, instead of generating commands one-by-one, the model outputs a massive **24x24 matrix** of dispatch percentages in a single forward pass. This guarantees ultra-fast inference and completely eliminates error cascades.

---

## 2. Feature Engineering Pipeline (`features.py`)

The Kaggle environment provides an `observation` dictionary containing lists of floating-point planets and fleets. Before the neural network can process this, it must be embedded into a continuous, fixed-size image tensor.

The board is represented as an Image Tensor of shape `(19, 64, 64)`.

### A. The 64x64 Spatial Grid
We divide the Orbit Wars map into a 64x64 pixel grid. This ensures that every planet occupies a unique pixel, allowing the Convolutional Neural Network (CNN) to inherently understand the 2D distances between planets, the path of incoming fleets, and cluster densities without needing complex mathematical heuristics.

### B. Planet & Fleet Channels (Channels 0-13)
These features are painted directly onto the exact `(X, Y)` grid coordinates where the planets or fleets exist:
1. **Ownership Masks**: 1-hot pixels for My Planets, Enemy Planets, and Neutrals.
2. **Economic Metrics**: Normalized production rates, ship garrisons, and radii.
3. **The Arrival Ledger**: We iterate through all in-flight fleets and calculate their exact ETA to every planet. We paint the "Future Friendly Ships" and "Future Enemy Ships" directly onto the destination planet's pixel.
4. **Active Fleets**: The actual $(X, Y)$ positions and Cosine/Sine trajectories of fleets currently flying through the void of space.

### C. Global Channels (Channels 14-18)
These describe macro-state metrics. Because a CNN cannot easily deduce global state from isolated pixels, these values are *broadcast uniformly* across the entire 64x64 image:
1. **Time**: `step / 500.0`.
2. **Resource Ratios**: My total planets vs Enemy total planets.
3. **The Sun Mask**: A circular mask drawn in the center of the grid representing the collision boundary of the Sun.

---

## 3. Data Extraction & Matrix Assembly (`extract.py`)

To train the model using Supervised Imitation Learning, we extract the "correct" dispatch matrices $Y$ from 20,000 top-tier replays.

### Ray-Sphere Target Interception
The raw JSON replay logs actions as `[Source_ID, Launch_Angle, Ship_Count]`. It does *not* log the intended target planet. `extract.py` casts a mathematical ray from the `Source` at `Launch_Angle` and calculates the perpendicular intersection to reverse-engineer the Target Planet.

### The 24x24 Label Matrix
Instead of a 1D sequence of actions, the label $Y$ is a `24x24` matrix.
- **Rows ($i$)**: The 24 Source planets.
- **Columns ($j$)**: The 24 Target planets.
- **Values**: Discretized bins representing the percentage of ships to send from $i$ to $j$ (e.g., 0%, 25%, 50%, 100%).

---

## 4. Neural Network Architecture (`model.py`)

The architecture is a Hybrid Convolutional Extractor with a Matrix Attention Head.

### A. The UNetEncoder
*   **Input**: $X \in \mathbb{R}^{B \times 19 \times 64 \times 64}$
*   **Mechanism**: A standard Computer Vision U-Net. It processes the grid through a series of Downsampling Convolutions (to extract macro-tactical context) and Upsampling Convolutions (to restore pixel-perfect accuracy). 
*   **Why U-Net?**: It natively understands 2D spatial relationships. It "sees" that Planet A is surrounded by Enemy Fleets.
*   **Output**: A rich spatial feature map $H \in \mathbb{R}^{B \times Hidden \times 64 \times 64}$.

### B. The UNetMatrixModel (Coordinate Plucking)
*   Instead of guessing actions for all 4,096 pixels, the model accepts the exact `(gx, gy)` grid coordinates of the 24 planets.
*   It "plucks" the CNN embeddings precisely at those 24 pixels: $E \in \mathbb{R}^{B \times 24 \times Hidden}$.
*   It passes these 24 embeddings through `Source` and `Target` linear projections.
*   It cross-multiplies the expanded Source and Target embeddings through a Multi-Layer Perceptron to output the final matrix.
*   **Output**: $Logits \in \mathbb{R}^{B \times 24 \times 24 \times NUM\_BINS}$.

---

## 5. The Training Pipeline (`train.py`)

Training utilizes **Masked 2D Cross-Entropy Loss**.

Since we output a 24x24 matrix, the model makes $24 \times 24 = 576$ predictions per turn. However, a player only owns a fraction of the 24 planets, and thus can only legally issue commands from their own planets.
We generate a `Valid_Mask` that zeroes out the loss for any row $i$ where the player does not own Planet $i$. This ensures the model is solely penalized for bad tactical dispatches from bases it actually controls, preventing it from hallucinating actions for enemy bases.

---

## 6. Live Inference & Physics Translation (`physics.py`)

During a live Kaggle match, inference is virtually instantaneous:
1. The `UNetEncoder` processes the board.
2. The `MatrixModel` outputs the 24x24 matrix.
3. We take `argmax(Logits)` to find the chosen dispatch fractions.
4. For any cell $i, j$ where the fraction > 0%, we trigger a command.

### The Deterministic Sandbox
Neural Networks are excellent at strategy, but terrible at floating-point trigonometry. To solve this, the ML model outputs **Intent** (Send 50% from Planet A to Planet B). The `agent.py` script unpacks these IDs and hands them to the `WorldModel` inside `physics.py`.

The `WorldModel.plan_shot(src_id, tgt_id)` function performs:
1. **Target Tracking**: If the target is an inner planet, it calculates its exact $(X, Y)$ position based on its angular velocity at the exact turn the fleet will intercept it.
2. **Sun Avoidance**: It calculates the perpendicular distance of the flight path to the Sun. If the distance is too close, it completely cancels the ML model's requested move, saving the fleet from destruction.
3. **Trig Translation**: It uses `math.atan2` to calculate the flawless, mathematically exact launch angle.

The combination of the Neural Network's macro-strategy and `physics.py`'s flawless micro-execution is the mechanism that breaks the 1000 Elo barrier.
