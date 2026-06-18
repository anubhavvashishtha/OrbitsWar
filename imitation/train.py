import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt

from model import UNetMatrixModel, N_CHANNELS

# Kaggle Environment Detection
if os.path.exists("/kaggle/working"):
    print("Kaggle environment detected! Redirecting I/O to /kaggle/working/...")
    DATASET_DIR = Path("/kaggle/working")
    MODELS_DIR = Path("/kaggle/working/models")
    LOGS_DIR = Path("/kaggle/working/logs")
else:
    DATASET_DIR = Path(__file__).parent / "chunks"
    MODELS_DIR = Path(__file__).parent / "models"
    LOGS_DIR = Path(__file__).parent / "logs"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "training_logs.json"
PLOT_FILE = LOGS_DIR / "training_loss_curve.png"

def plot_logs(logs):
    if not logs:
        return
    epochs = [log['epoch'] for log in logs]
    losses = [log['loss'] for log in logs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('UNet Matrix Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Masked 2D CrossEntropy Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_FILE)
    plt.close()

def train(epochs=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = UNetMatrixModel(in_channels=N_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. Load latest model and logs
    latest_pt = MODELS_DIR / "unet_matrix_latest.pt"
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
    
    start_epoch = logs[-1]['epoch'] + 1 if logs else 1
    
    chunk_files = sorted(list(DATASET_DIR.glob("dataset_chunk_*.npz")))
    if not chunk_files:
        print(f"No dataset chunks found in {DATASET_DIR}")
        return
        
    print(f"Found {len(chunk_files)} dataset chunks to train on.")
    
    import random
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        total_valid_entries = 0
        
        # Shuffle the order of chunk files every epoch for maximum generalization!
        random.shuffle(chunk_files)
        
        for chunk_file in chunk_files:
            data = np.load(chunk_file)
            X = torch.from_numpy(data['X']).float()
            C = torch.from_numpy(data['Coords']).float()
            Y = torch.from_numpy(data['Y']).long()
            M = torch.from_numpy(data['Mask']).bool()
            
            dataset = TensorDataset(X, C, Y, M)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for x_batch, c_batch, y_batch, mask_batch in dataloader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                optimizer.zero_grad()
                
                # logits: (B, 24, 24, NUM_BINS)
                logits = model(x_batch, c_batch)
                
                # CrossEntropy requires (B, C, d1, d2)
                logits = logits.permute(0, 3, 1, 2)  # (B, NUM_BINS, 24, 24)
                
                # Calculate raw loss: (B, 24, 24)
                loss_matrix = F.cross_entropy(logits, y_batch, reduction='none')
                
                # Mask out invalid rows (Planets we don't own)
                # mask_batch: (B, 24). Expand to (B, 24, 24)
                expanded_mask = mask_batch.unsqueeze(2).expand_as(loss_matrix).float()
                
                masked_loss = loss_matrix * expanded_mask
                
                # Average loss only over valid planet rows
                valid_count = expanded_mask.sum()
                if valid_count > 0:
                    loss = masked_loss.sum() / valid_count
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += masked_loss.sum().item()
                    total_valid_entries += valid_count.item()
                    
        if total_valid_entries > 0:
            avg_loss = total_loss / total_valid_entries
            print(f"Epoch {epoch}/{start_epoch + epochs - 1} | Loss: {avg_loss:.4f}")
            
            logs.append({'epoch': epoch, 'loss': avg_loss})
            
            # Save checkpoints
            torch.save(model.state_dict(), MODELS_DIR / "unet_matrix_latest.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), MODELS_DIR / "unet_matrix_best.pt")
                print(" -> New best model saved!")
                
            with open(LOG_FILE, 'w') as f:
                json.dump(logs, f)
            plot_logs(logs)

if __name__ == "__main__":
    train(epochs=30)
