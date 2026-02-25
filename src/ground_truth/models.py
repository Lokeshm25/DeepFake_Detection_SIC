# src/ground_truth/models.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import timm


# ---------------------------
# Head-only module (works on cached embeddings)
# ---------------------------
class SpatialHead(nn.Module):
    def __init__(self, feat_dim, head_hidden=512, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, feats):
        # feats: [N, feat_dim] -> returns [N]
        return self.head(feats).squeeze(-1)


# ---------------------------------------------------------
# SPATIAL MODEL (EfficientNet-B3 + Custom Head)
# - Optional: construct head-only (when feat_dim is provided),
#   or full backbone+head when using real images.
# ---------------------------------------------------------
class SpatialModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b3", pretrained=False, head_hidden=512, dropout=0.4, feat_dim: int = None):
        """
        If feat_dim is provided, we *do not* instantiate a timm backbone and
        treat this module as a head-only model that accepts feature vectors of size feat_dim.
        Otherwise, we create timm backbone and infer feat_dim from backbone.num_features.
        """
        super().__init__()
        self.backbone = None
        if feat_dim is None:
            # create backbone and infer feat dim
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            feat_dim = self.backbone.num_features

        self.feat_dim = feat_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        """
        If backbone exists: x is image tensor [B, C, H, W] -> backbone -> head.
        If backbone is None: x is expected to be feature vectors [B, feat_dim] -> head.
        """
        if self.backbone is not None:
            feats = self.backbone(x)
        else:
            feats = x
        logits = self.head(feats).squeeze(-1)
        return logits

    def forward_head(self, feats):
        """Features -> Head -> Logits (Useful for ensemble phase using cached embeddings)"""
        return self.head(feats).squeeze(-1)


# ---------------------------------------------------------
# ATTENTION POOL
# ---------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, h, lengths):
        # h: [B, T, H]
        B, T, _ = h.shape
        # ensure lengths is on same device
        lengths = lengths.to(h.device)
        mask = torch.arange(T, device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = self.att(h).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask, float("-1e9"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        out = (h * weights.unsqueeze(-1)).sum(dim=1)
        return out, weights


# ---------------------------------------------------------
# TEMPORAL MODEL (LSTM + Attention Pooling)
# ---------------------------------------------------------
class TemporalModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512, n_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = AttentionPool(out_dim)

        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, lengths):
        # x: [B, T, feat], lengths: [B] (long tensor)
        if lengths.numel() == 0:
            raise ValueError("Empty lengths in TemporalModel")

        # move lengths to device of x
        lengths = lengths.to(x.device)

        # Sort for packing (LSTM requirement)
        lengths_sorted, perm_idx = lengths.sort(descending=True)
        x_sorted = x[perm_idx]

        # Pack (lengths must be on CPU for pack_padded_sequence)
        packed = rnn_utils.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM
        packed_out, _ = self.lstm(packed)

        # Unpack
        out_unpacked, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

        # Unsort back to original order
        _, unperm_idx = perm_idx.sort()
        out = out_unpacked[unperm_idx]
        lengths = lengths[unperm_idx]

        # Attention Pooling
        pooled, _ = self.attn(out, lengths)

        # Head
        logits = self.head(pooled).squeeze(-1)
        return logits