# Orbit Wars - Training Guide

This guide explains how to train both of your bots: the **Classifier Bot** (Gradient Boosting) and the **RL PPO Actor-Critic Bot**.

---

## 1. The Classifier Bot (Gradient Boosting)
*This is your primary, highest-performing bot. It uses Imitation Learning to memorize and copy expert strategies from replays.*

### Step 1: Harvest Replays
1. Go to the Kaggle Orbit Wars Leaderboard.
2. Find matches where top players (or yourself) performed well.
3. Download the `.json` replay files.
4. Move all downloaded `.json` files into the `records/` folder in your project.

### Step 2: Train the Brain
1. Open your terminal in the `OrbitsWar` project folder.
2. Run the replay training script:
   ```bash
   python classifier/train_from_replays.py
   ```
3. The script will extract all the board states, balance the wins and losses, and train the `value_fn.py` model. 

### Step 3: Package for Submission
1. Once training is complete, your `value_fn.py` is automatically updated.
2. Run your submission script:
   ```bash
   python submission_script.py
   ```
3. Upload `submission_classifier.zip` to Kaggle!

---

## 2. The RL PPO Actor-Critic Bot
*This is your experimental deep reinforcement learning bot. It learns via a Neural Network through a mix of replay observation and self-play trial-and-error.*

### Step 1: Harvest Replays (Pre-training)
Just like the classifier, the RL bot benefits from looking at Kaggle replays first so it doesn't have to learn how to play from total scratch.
1. Ensure your downloaded `.json` replays are in the `records/` folder.

### Step 2: Run the Pipeline
The RL bot has a unified training pipeline that handles Pre-training, Self-play, and Exporting automatically.

1. Open your terminal in the `OrbitsWar` project folder.
2. Run the PPO Agent pipeline:
   ```bash
   python rl/ppo_agent.py
   ```
3. **Phase 1:** It will parse the replays in `records/` to pre-train the neural network (`rl/model.pt`).
4. **Phase 2:** It will begin Self-Play rounds to fine-tune the model against enemy bots. This phase takes the longest.
5. **Phase 3:** Once self-play is done, it will automatically export its weights to `rl_value_fn.py`.

*Note: If you want to skip the replay phase and just do self-play, run:* `python rl/ppo_agent.py --skip-replay`

### Step 3: Package for Submission
1. Once the pipeline finishes and `rl_value_fn.py` is updated, run the submission script:
   ```bash
   python submission_script.py
   ```
2. Upload `submission_rl.zip` to Kaggle!
