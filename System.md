# Orbit Wars: Autoregressive Seq2Seq Architecture Whitepaper

This document is the comprehensive architectural whitepaper for the Orbit Wars Machine Learning bot. It details the exact mathematical transformations, tensor dimensions, neural network architecture, and data pipelines required to translate a 100x100 real-time strategy grid into a sequence of tokenized fleet commands.

---

## 1. The Core Paradigm Shift

### The Problem with Edge-Scoring
In typical RL or Imitation Learning for graph-based games, agents are trained to evaluate edges. Given Planet A and Planet B, what is the probability $P(attack)$? 
This fails in Orbit Wars because the game requires **Swarm Coordination**. If an enemy planet has 100 ships, sending 50 ships from Base A and 50 ships from Base B *at the exact same time* is required to capture it. Independent edge scoring cannot coordinate two separate bases to attack the same target simultaneously without splitting the budget erroneously.

### The Autoregressive Solution
We reformulate Orbit Wars as a **Language Modeling Problem**. Instead of evaluating edges, the model generates a "sentence" of commands for the current turn.
By predicting tokens sequentially with a Causal Mask, the model's $N_{th}$ decision is mathematically conditioned on its $(N-1)_{th}$ decision. It natively learns that if it just ordered an attack on Planet X using Base A, but Planet X is still heavily defended, its *very next token* should be an attack on Planet X using Base B.

---

## 2. Feature Engineering Pipeline (`features.py`)

The Kaggle environment provides an `observation` dictionary containing lists of planets and fleets. Before the neural network can process this, it must be embedded into a continuous, fixed-size tensor.

The board is represented as a Tensor of shape `(MAX_PLANETS, 55)`, where $MAX\_PLANETS = 200$.

### Global Features (31 Dimensions)
These features describe the macro-state of the game and are appended identically to every planet's vector.
1. **Time**: `step / 500.0` (1D)
2. **Resource Ownership**: The total ships and production capacity owned by Player 1, Player 2, Player 3, Player 4, and Neutrals. Normalized against the total ships on the board. (10D)
3. **Fleet Demographics**: The number of friendly fleets currently in transit vs enemy fleets. (2D)
4. **Spatial Clusters**: The center-of-mass (X, Y) of your empire vs the enemies' empires. (4D)
... and other macro metrics totaling 31 dimensions.

### Planet Features (24 Dimensions)
These features are calculated specifically for Planet $i$, providing a localized, ego-centric perspective.
1. **Spatial Coordinates**: Normalized $X$ and $Y$ positions, radius, and angular velocity (for inner rotating planets).
2. **Economic Metrics**: Production rate, current garrison size (normalized).
3. **Ownership Encoding**: One-hot encoding of whether the planet is owned by Us, Enemy 1, Enemy 2, Enemy 3, or Neutral.
4. **The Arrival Ledger**: This is the most critical feature. The engine iterates through all fleets currently in flight, calculates their ETA to Planet $i$, and builds a timeline. The features encode:
   - Total friendly ships arriving in the future.
   - Total enemy ships arriving in the future.
   - Turns until the *first* friendly arrival.
   - Turns until the *first* enemy arrival.
   This gives the model a sense of the "future" state of the planet, allowing it to send reinforcements *before* an enemy fleet lands.

---

## 3. Data Extraction & Tokenization (`extract.py`)

To train the model using Supervised Imitation Learning, we must extract the "correct" token sequence $Y$ from the 23,000 top-tier replays.

### Ray-Sphere Target Interception
The raw JSON replay logs actions as `[Source_ID, Launch_Angle, Ship_Count]`. It does *not* log the intended target planet. 
To reverse-engineer the target, `extract.py` casts a mathematical ray from the `Source` at `Launch_Angle`. It calculates the perpendicular distance from this ray to the center of every other planet. The planet that intersects the ray (perpendicular distance $< Radius + 1$) is registered as the Target.

### Vocabulary Layout
The model predicts discrete integer tokens. The vocabulary size is $V = 212$.
- **Tokens `0` to `199`**: Planet IDs. (Used for both Targets and Sources).
- **Tokens `200` to `209`**: Ship Fractions. (Bin 0 = 10%, Bin 9 = 100%).
- **Token `210`**: `<END_TURN>`
- **Token `211`**: `<START>`

### Sequence Construction
For a single turn, the winner's actions are sorted by `Ship_Count` descending. The sequence is constructed:
$Y = [211, Tgt_1, Src_1, Frac_1, Tgt_2, Src_2, Frac_2, 210, 210, 210...]$
The sequence is padded to `MAX_SEQ_LEN = 60` using the `<END_TURN>` token.

---

## 4. Neural Network Architecture (`model.py`)

The `AutoregressiveOrbitModel` is built purely on PyTorch Transformers.

### A. The OrbitEncoder (Self-Attention)
*   **Input**: $X_{board} \in \mathbb{R}^{B \times 200 \times 55}$
*   **Projection**: A linear layer projects the 55-D features into a hidden dimension $D_{model} = 128$.
*   **Mechanism**: A standard TransformerEncoder (4 layers, 4 attention heads). 
*   **Why Self-Attention?**: Gravity boards are fully connected graphs. A planet on the left edge of the map mathematically influences a planet on the right edge. Self-Attention allows $Planet_A$ to query the feature vector of $Planet_B$ regardless of spatial distance, building a holistic `Memory` tensor representing the tactical state of the entire galaxy.
*   **Output**: $Memory \in \mathbb{R}^{B \times 200 \times 128}$

### B. The OrbitDecoder (Causal Cross-Attention)
*   **Input**: The previously generated sequence of tokens $Y_{prev} \in \mathbb{Z}^{B \times S}$.
*   **Embedding**: An `nn.Embedding` layer converts the discrete tokens (0-211) into 128-D continuous vectors.
*   **Positional Encoding**: Sinusoidal positional embeddings are added so the model knows the order of the sequence.
*   **Mechanism**: A TransformerDecoder (4 layers, 4 attention heads).
    *   **Causal Masking**: A lower-triangular mask ensures that when predicting the token at step $t$, the self-attention mechanism can only look at tokens $0$ through $t-1$. It cannot "cheat" by looking into the future.
    *   **Cross-Attention**: The decoder queries the $Memory$ tensor generated by the Encoder to ground its decisions in the physical reality of the board state.
*   **Output Head**: A final linear layer projects the 128-D decoder output into $Logits \in \mathbb{R}^{B \times S \times 212}$.

---

## 5. The Training Pipeline (`train.py`)

Training utilizes **Teacher Forcing** and **Causal Cross-Entropy Loss**.

### Data Shifting
Given a padded target sequence:
$Y = [START, Tgt_1, Src_1, Frac_1, END, END, END]$

We split this into Input and Target:
$Input_{seq} = Y[:, :-1] = [START, Tgt_1, Src_1, Frac_1, END, END]$
$Target_{seq} = Y[:, 1:] = [Tgt_1, Src_1, Frac_1, END, END, END]$

### Loss Masking
We compute `CrossEntropyLoss` between the Decoder's predicted Logits and the $Target_{seq}$. 
Crucially, we must teach the model to output the *first* `<END>` token, but we do not want to penalize it for the padded `<END>` tokens that follow.
We create a `valid_mask = (Input_{seq} != END_TOKEN)`. 
This allows the loss gradient to flow through the prediction of the first `<END>` token, but zeroes out the loss for all subsequent padding tokens.

---

## 6. Live Inference & Physics Translation (`agent.py` & `physics.py`)

In a live Kaggle match, the agent cannot use Teacher Forcing because there is no teacher. It must decode **Greedily**.

### Greedy Decoding Loop
1. The Encoder processes the board and generates the $Memory$ tensor.
2. The sequence $Y$ is initialized as `[[START]]`.
3. The Decoder processes $Y$ and outputs Logits.
4. We take `argmax(Logits)` to find the most probable next token.
5. We append this token to $Y$.
6. We repeat steps 3-5 until the model outputs `<END>`.

### The Deterministic Sandbox (`physics.py`)
Neural Networks are excellent at strategy, but terrible at floating-point trigonometry. If the ML model directly output a launch angle of `1.5707`, it might be off by `0.0001` radians, causing the fleet to miss the target or graze the Sun's collision radius and instantly vaporize.

To solve this, the ML model outputs **Intent** (Discrete Tokens: Source ID, Target ID). 
The `agent.py` script unpacks these IDs and hands them to the `WorldModel` inside `physics.py`.

The `WorldModel.plan_shot(src_id, tgt_id)` function performs:
1. **Target Tracking**: If the target is an inner planet, it calculates its exact $(X, Y)$ position based on its angular velocity at the exact turn the fleet will intercept it.
2. **Sun Avoidance**: It calculates the perpendicular distance of the flight path to the center of the board $(0,0)$. If the distance is less than the Sun's radius, it completely cancels the ML model's requested move, saving the fleet from destruction.
3. **Trig Translation**: It uses `math.atan2` to calculate the flawless, mathematically exact launch angle.

The combination of the Neural Network's macro-strategy and `physics.py`'s flawless micro-execution is the mechanism that breaks the 1000 Elo barrier.
