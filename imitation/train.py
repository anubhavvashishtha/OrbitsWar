import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt

from model import AutoregressiveOrbitModel, END_TOKEN, START_TOKEN, MAX_PLANETS

DATASET_PATH = Path(__file__).parent / "dataset.npz"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_FILE = CHECKPOINT_DIR / "training_logs.json"
PLOT_FILE = CHECKPOINT_DIR / "training_loss_curve.png"

def plot_logs(logs):
    if not logs:
        return
    epochs = [log['epoch'] for log in logs]
    losses = [log['loss'] for log in logs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Autoregressive Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average CrossEntropy Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_FILE)
    plt.close()

def train(epochs=10, batch_size=32):
    print("Loading dataset...")
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        return
        
    data = np.load(DATASET_PATH)
    X = torch.from_numpy(data['X']).float()
    Mask = torch.from_numpy(data['Mask']).bool()
    Y = torch.from_numpy(data['Y']).long()
    
    print(f"Dataset size: {len(X)} steps")
    
    dataset = TensorDataset(X, Mask, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = AutoregressiveOrbitModel(in_features=55).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # 1. Load latest model and logs
    latest_pt = CHECKPOINT_DIR / "autoregressive_latest.pt"
    if latest_pt.exists():
        print(f"Resuming from latest checkpoint: {latest_pt}")
        model.load_state_dict(torch.load(latest_pt, map_location=device))
        
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
            print(f"Loaded {len(logs)} past epochs from logs.")
        except Exception as e:
            print(f"Error loading logs: {e}")
            
    best_loss = float('inf')
    if logs:
        best_loss = min(log['loss'] for log in logs)
    
    start_epoch = logs[-1]['epoch'] if logs else 0
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (x_batch, mask_batch, y_batch) in enumerate(dataloader):
            x_batch, mask_batch, y_batch = x_batch.to(device), mask_batch.to(device), y_batch.to(device)
            
            tgt_input = y_batch[:, :-1]
            tgt_target = y_batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(x_batch, tgt_input, src_mask=mask_batch)
            
            B, S, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            target_flat = tgt_target.reshape(-1)
            
            loss = criterion(logits_flat, target_flat).reshape(B, S)
            
            valid_mask = (tgt_input != END_TOKEN).float()
            masked_loss = (loss * valid_mask).sum()
            tokens_in_batch = valid_mask.sum()
            
            batch_loss = masked_loss / tokens_in_batch
            batch_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            total_tokens += tokens_in_batch.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {batch_loss.item():.4f}")
                
        avg_loss = total_loss / total_tokens
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")
        
        # 2. Append logs and write to JSON
        logs.append({"epoch": epoch + 1, "loss": avg_loss})
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)
            
        # 3. Create graph
        plot_logs(logs)
        
        # 4. Save latest and best models
        torch.save(model.state_dict(), latest_pt)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_pt = CHECKPOINT_DIR / "autoregressive_best.pt"
            torch.save(model.state_dict(), best_pt)
            print(f"New best model saved! (Loss: {best_loss:.4f})")
            
    print("Training run complete.")

if __name__ == "__main__":
    train(epochs=50)
