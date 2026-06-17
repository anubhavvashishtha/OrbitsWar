import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PLANETS = 200
NUM_FRACTION_BINS = 10
VOCAB_SIZE = MAX_PLANETS + NUM_FRACTION_BINS + 2
END_TOKEN = VOCAB_SIZE - 2
START_TOKEN = VOCAB_SIZE - 1

class OrbitEncoder(nn.Module):
    def __init__(self, in_features, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        # x: (B, P, in_features)
        x = self.input_proj(x)
        # mask is boolean (B, P) where True means padded planet
        if mask is not None:
            memory = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            memory = self.transformer_encoder(x)
        return memory # (B, P, d_model)

class OrbitDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_encoder = nn.Embedding(500, d_model) # max sequence length per turn
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, VOCAB_SIZE)
        
    def forward(self, tgt_tokens, memory, tgt_mask=None, memory_key_padding_mask=None):
        # tgt_tokens: (B, S) - sequence of tokens
        # memory: (B, P, d_model) - output from encoder
        
        B, S = tgt_tokens.shape
        positions = torch.arange(S, device=tgt_tokens.device).unsqueeze(0).expand(B, S)
        
        x = self.token_embedding(tgt_tokens) + self.pos_encoder(positions)
        
        if tgt_mask is None:
            # causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=tgt_tokens.device)
            
        out = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        logits = self.output_proj(out) # (B, S, VOCAB_SIZE)
        return logits

class AutoregressiveOrbitModel(nn.Module):
    def __init__(self, in_features=31, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.encoder = OrbitEncoder(in_features, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.decoder = OrbitDecoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        
    def forward(self, x, tgt_tokens, src_mask=None):
        # For training
        memory = self.encoder(x, mask=src_mask)
        logits = self.decoder(tgt_tokens, memory, memory_key_padding_mask=src_mask)
        return logits

    @torch.no_grad()
    def generate(self, x, max_len=60, src_mask=None):
        # For inference (batch size 1)
        self.eval()
        memory = self.encoder(x, mask=src_mask)
        
        B = x.size(0)
        device = x.device
        
        # Start with START_TOKEN
        generated = torch.full((B, 1), START_TOKEN, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            logits = self.decoder(generated, memory, memory_key_padding_mask=src_mask)
            # Take the logits for the last step
            next_token_logits = logits[:, -1, :]
            
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # If all batches have produced END_TOKEN, stop
            if (generated == END_TOKEN).any(dim=1).all():
                break
                
        return generated
