import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_FRACTION_BINS = 6  # e.g., 0%, 20%, 40%, 60%, 80%, 100%
N_CHANNELS = 19

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        # 1x1 conv to match dimensions if channels change
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        # Down
        self.inc = ResidualBlock(in_channels, hidden_dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hidden_dim, hidden_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hidden_dim * 2, hidden_dim * 4))
        
        # Bottleneck (Residual network between downward and upward)
        self.bottleneck = nn.Sequential(
            ResidualBlock(hidden_dim * 4, hidden_dim * 4),
            ResidualBlock(hidden_dim * 4, hidden_dim * 4)
        )
        
        # Up
        self.up1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2)
        self.conv_up1 = ResidualBlock(hidden_dim * 4, hidden_dim * 2)
        
        self.up2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.conv_up2 = ResidualBlock(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: (B, 19, 64, 64)
        x1 = self.inc(x)             # (B, 64, 64, 64)
        x2 = self.down1(x1)          # (B, 128, 32, 32)
        x3 = self.down2(x2)          # (B, 256, 16, 16)
        
        # Pass through the bottleneck residual network
        x_bot = self.bottleneck(x3)  # (B, 256, 16, 16)
        
        x = self.up1(x_bot)          # (B, 128, 32, 32)
        x = torch.cat([x, x2], dim=1) # (B, 256, 32, 32)
        x = self.conv_up1(x)         # (B, 128, 32, 32)
        
        x = self.up2(x)              # (B, 64, 64, 64)
        x = torch.cat([x, x1], dim=1) # (B, 128, 64, 64)
        x = self.conv_up2(x)         # (B, 64, 64, 64)
        return x

class UNetMatrixModel(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, hidden_dim=64, num_bins=NUM_FRACTION_BINS):
        super().__init__()
        self.unet = UNetEncoder(in_channels, hidden_dim)
        
        # Matrix Head
        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tgt_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.matrix_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, spatial_map, planet_coords):
        """
        spatial_map: (Batch, N_CHANNELS, 64, 64)
        planet_coords: (Batch, 24, 2) containing (gx, gy) for each planet.
        """
        batch_size = spatial_map.size(0)
        num_planets = planet_coords.size(1)
        
        # 1. Run Spatial Feature Extraction
        feature_map = self.unet(spatial_map)  # (B, Hidden, 64, 64)
        
        # 2. Pluck Embeddings
        # Extract features at the exact (gx, gy) coordinates
        # planet_coords are (gx, gy). We need to index dim 2 (Y) and dim 3 (X).
        b_idx = torch.arange(batch_size).view(batch_size, 1).expand(-1, num_planets)
        gx = planet_coords[:, :, 0].long()
        gy = planet_coords[:, :, 1].long()
        
        # feature_map shape is (B, C, H, W). We want (B, N, C).
        # We index: feature_map[b_idx, :, gy, gx]
        # To do this safely, we permute to (B, H, W, C) first.
        feature_map = feature_map.permute(0, 2, 3, 1)  # (B, 64, 64, Hidden)
        planet_embeddings = feature_map[b_idx, gy, gx, :]  # (B, 24, Hidden)
        
        # 3. Create Matrix Pairings
        src_feats = self.src_proj(planet_embeddings)  # (B, 24, Hidden)
        tgt_feats = self.tgt_proj(planet_embeddings)  # (B, 24, Hidden)
        
        # Expand and concat
        # src_feats: (B, 24, 1, Hidden) -> expand to (B, 24, 24, Hidden)
        # tgt_feats: (B, 1, 24, Hidden) -> expand to (B, 24, 24, Hidden)
        src_exp = src_feats.unsqueeze(2).expand(batch_size, num_planets, num_planets, -1)
        tgt_exp = tgt_feats.unsqueeze(1).expand(batch_size, num_planets, num_planets, -1)
        
        pair_feats = torch.cat([src_exp, tgt_exp], dim=-1)  # (B, 24, 24, Hidden * 2)
        
        # 4. Predict Dispatches
        matrix_out = self.matrix_mlp(pair_feats)  # (B, 24, 24, NUM_BINS)
        
        return matrix_out
