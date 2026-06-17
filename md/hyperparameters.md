# Orbit Wars: Hyperparameters Guide

The Autoregressive Neural Network is configured by several hyperparameters. Tweaking these can drastically alter how fast the model learns, how well it generalizes, and how much VRAM it consumes.

Here is a breakdown of the key hyperparameters in this system, where to find them, and how they affect your bot.

---

## 1. Sequence Length (`MAX_SEQ_LEN`)
**Location:** `imitation/extract.py` (Line ~16)
**Current Value:** `60`

* **What it does:** This is the maximum number of "tokens" the model is allowed to predict per turn. Since each action consists of 3 tokens (`Target`, `Source`, `Fraction`), a length of `60` means the model can issue up to 19 distinct fleet commands in a single turn.
* **Effect of changing:**
  * If you set it to `10`: The model will only be able to issue 3 commands per turn. It will learn much faster, but it won't be able to coordinate massive, multi-planet swarm attacks.
  * If you set it to `150`: The model can issue 50 commands per turn. However, the sequence gets very long, meaning the Causal Masking takes exponentially more VRAM to process, and the model might struggle to remember the beginning of its turn.

## 2. Learning Rate (`lr`)
**Location:** `imitation/train.py` (Line ~37, inside the `Adam` optimizer)
**Current Value:** `1e-4` (0.0001)

* **What it does:** Controls how drastically the neural network updates its internal weights after every batch.
* **Effect of changing:**
  * **Higher (`1e-3` / 0.001):** The model learns *much* faster, but risks "overshooting" the optimal strategy and becoming unstable (the loss might bounce up and down violently instead of dropping smoothly).
  * **Lower (`1e-5` / 0.00001):** Extremely stable, smooth learning, but will require hundreds of epochs and hours of computation to converge.

## 3. Batch Size (`batch_size`)
**Location:** `imitation/train.py` (Line ~16, argument to `train()`)
**Current Value:** `32`

* **What it does:** The number of turns (sequences) the model looks at simultaneously before updating its weights. 
* **Effect of changing:**
  * **Higher (`128` or `256`):** Training runs much faster because GPUs love parallel matrix math. The gradient updates are smoother. However, if your GPU doesn't have enough VRAM, you will get an `Out of Memory` (OOM) error.
  * **Lower (`8` or `16`):** Training takes longer, and the loss graph will look "noisier" because the model is updating its weights based on a very small sample of moves.

## 4. Epochs (`epochs`)
**Location:** `imitation/train.py` (Line ~16, argument to `train()`)
**Current Value:** `50` (Default for full training)

* **What it does:** How many times the model reads the *entire* dataset.
* **Effect of changing:**
  * If the model loss is still dropping smoothly at the end of training, you didn't train for enough epochs. You should run the script again (it will load the latest weights automatically) and train for more epochs.
  * If the loss hits `1.5` and stays perfectly flat for 10 epochs, you have trained enough. Further training won't help.

## 5. Fraction Bins (`NUM_FRACTION_BINS`)
**Location:** `imitation/extract.py` & `imitation/model.py`
**Current Value:** `10`

* **What it does:** The model cannot output continuous floating-point numbers natively. Instead, we divide `0%` to `100%` into discrete categories. `10` bins means the model can output 10%, 20%, 30%, etc.
* **Effect of changing:**
  * **Setting to `100`**: The model could output exactly 47% or 93%. But the Vocabulary size expands massively, making the prediction task 10x harder.
  * **Setting to `5`**: The model can only output 20%, 40%, 60%, 80%, 100%. It learns very fast, but lacks fine-grained tactical control over its fleets. `10` is the perfect middle ground for Orbit Wars.
