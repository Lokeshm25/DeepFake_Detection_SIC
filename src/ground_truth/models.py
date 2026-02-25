import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import timm

# ---------------------------------------------------------
# SPATIAL MODEL (EfficientNet-B3 + Custom Head)
# Matches train_spatial.txt
# ---------------------------------------------------------
class SpatialModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b3", pretrained=True, head_hidden=512, dropout=0.4):
        super().__init__()
        # Create backbone (num_classes=0 returns feature vector)
        # If pretrained=False, it avoids downloading weights (useful for inference/ensemble)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        
        # Standard classification head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1) # logits
        )

    def forward(self, x):
        """Full forward pass: Image -> Backbone -> Head -> Logits"""
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(1)
        return logits

    def forward_features(self, x):
        """Image -> Backbone -> Features (Size: [B, feat_dim])"""
        return self.backbone(x)

    def forward_head(self, feats):
        """Features -> Head -> Logits (Useful for ensemble phase using cached embeddings)"""
        return self.head(feats).squeeze(1)


# ---------------------------------------------------------
# TEMPORAL MODEL (LSTM + Attention Pooling)
# Matches train_temporal.txt
# ---------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, h, lengths):
        # h: [B, T, H]
        B, T, _ = h.shape
        # Masking: create mask where (seq_idx >= length)
        mask = torch.arange(T, device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        scores = self.att(h).squeeze(-1) # [B, T]
        scores = scores.masked_fill(mask, -1e9)
        
        weights = torch.softmax(scores, dim=1)
        # Safety for NaNs
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        out = (h * weights.unsqueeze(-1)).sum(dim=1)
        return out, weights


class TemporalModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512, n_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0
        )
        
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = AttentionPool(out_dim)
        
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, lengths):
        # x: [B, T, feat], lengths: [B]
        
        if lengths.numel() == 0:
             raise ValueError("Empty lengths in TemporalModel")

        # Sort for packing (LSTM requirement)
        lengths_sorted, perm_idx = lengths.sort(descending=True)
        x_sorted = x[perm_idx]

        # Pack
        packed = rnn_utils.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM
        packed_out, _ = self.lstm(packed)
        
        # Unpack
        out_unpacked, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        
        # Unsort (restore original order)
        _, unperm_idx = perm_idx.sort()
        out = out_unpacked[unperm_idx]
        lengths = lengths[unperm_idx]
        
        # Attention Pooling
        pooled, _ = self.attn(out, lengths)
        
        # Classification Head
        logits = self.head(pooled).squeeze(1)
        
        return logits