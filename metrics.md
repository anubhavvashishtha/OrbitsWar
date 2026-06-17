# Orbit Wars: Metrics & Evaluation

When you run `train.py`, the system logs the training metrics to `imitation/checkpoints/training_logs.json` and automatically plots them to `imitation/checkpoints/training_loss_curve.png`.

Here is exactly what those metrics indicate and how to evaluate your model's performance.

## The Metric: Average CrossEntropy Loss

The only metric printed during training is **Loss** (specifically, Causal Cross-Entropy Loss). 

Unlike a simple "Accuracy %" metric (which is misleading for sequence generation), CrossEntropy Loss measures **how confident and correct the model is at predicting the next token** in the sequence.

### What the numbers mean:
*   **Loss > 5.0**: The model is guessing randomly. It hasn't learned the vocabulary.
*   **Loss ~ 3.5**: The model has learned the basic structure (e.g., that it must output `Target -> Source -> Fraction`), but is still guessing which planets to actually pick.
*   **Loss < 2.5**: The model is actively learning the strategy. It is correctly identifying the exact targets and sources that the top Kaggle bots use.
*   **Loss < 1.0**: The model is highly confident and almost perfectly mimicking the training dataset. *(Note: Getting too close to 0.0 means you are "overfitting" and the model might memorize the replays rather than learning the general strategy. A final loss between `1.0` and `2.0` on a massive dataset is usually the sweet spot!)*

### How to Evaluate Performance
1. Open `imitation/checkpoints/training_loss_curve.png`.
2. You want to see the blue line sloping downward smoothly.
3. If the line flattens out (stops decreasing) for many epochs, your model has "converged" and finished learning everything it can from the dataset.

---

## FAQ: Can this architecture learn by fighting enemies?

**Short Answer:** No, not natively.

**In-Depth Answer:** 
The Autoregressive model is designed strictly for **Imitation Learning (Supervised Learning)**. It requires a "Teacher" (the JSON replays) to tell it exactly what the correct sequence of tokens should be. It calculates its Loss by comparing its output to the Teacher's output.

If you put it in a live arena to fight an enemy without a teacher, it has no `Y_target` sequence to compare its answers to, so it cannot calculate its Loss and cannot update its weights.

### How *could* you make it learn by fighting?
To make it learn by fighting, you would have to wrap this entire architecture in a **Reinforcement Learning (PPO)** loop. 
1. You would let it play a game.
2. If it wins, you assign a `+1` reward to every single token it generated during that game.
3. If it loses, you assign a `-1` reward.

*However, applying PPO to autoregressive token sequences is notoriously difficult and computationally expensive (this is exactly what OpenAI does with RLHF for ChatGPT).* For Kaggle competitions like Orbit Wars, Imitation Learning on the top leaderboard replays is significantly faster, easier to tune, and usually yields a higher score than fighting from scratch!
