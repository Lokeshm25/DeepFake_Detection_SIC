# 06 - Train temporal model (LSTM + attention) on embeddings
# Video-level classifier using per-frame embeddings
# Saves checkpoints: checkpoints/temporal/

from pathlib import Path
import json, time, random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ---------------- USER CONFIG ----------------
ROOT = Path.cwd().parent
EMB_ROOT = ROOT / "embeddings"
LABELS_JSON = ROOT / "data" / "labels.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "temporal"

NUM_EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4            # ← you can safely use 4 or 6 now
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTM_HIDDEN = 512
LSTM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
PRINT_FREQ = 20
SEED = 42
# ------------------------------------------------


# ---------------- Utilities ----------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_auc(y_true, y_pred):
    if np.isnan(y_pred).any():
        return float("nan")
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return float("nan")
# -------------------------------------------


# ---------------- Dataset -------------------
with open(LABELS_JSON, "r") as f:
    labels_map = json.load(f)

def get_label_from_stem(stem):
    if stem in labels_map:
        return int(labels_map[stem])
    for k, v in labels_map.items():
        if stem in k:
            return int(v)
    raise KeyError(stem)

class VideoEmbeddingDataset(Dataset):
    def __init__(self, split):
        self.root = EMB_ROOT / split
        self.items = sorted(self.root.glob("*.npy"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        emb = np.load(p).astype(np.float32)     # [T, feat]
        label = get_label_from_stem(p.stem)
        return torch.from_numpy(emb), torch.tensor(label, dtype=torch.float32), p.stem

def collate_fn(batch):
    seqs, labels, stems = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    maxlen = max(lengths)
    feat_dim = seqs[0].shape[1]

    out = torch.zeros(len(seqs), maxlen, feat_dim)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s

    return out, lengths, torch.stack(labels), list(stems)
# -------------------------------------------


# ---------------- Model ---------------------
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        lengths = torch.clamp(lengths, min=1)

        scores = self.att(x).squeeze(-1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)

        w = torch.softmax(scores, dim=1)
        out = (x * w.unsqueeze(-1)).sum(dim=1)
        return out, w

class TemporalModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, LSTM_HIDDEN, LSTM_LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0
        )
        out_dim = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)
        self.attn = AttentionPool(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, lengths):
        lengths_sorted, idx = lengths.sort(descending=True)
        x_sorted = x[idx]

        packed = rnn_utils.pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(),
            batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

        _, inv = idx.sort()
        out = out[inv]
        lengths = lengths[inv]

        pooled, att = self.attn(out, lengths)
        logits = self.head(pooled).squeeze(1)
        return logits, att
# -------------------------------------------


# ---------------- Training ------------------
def main():
    set_seed()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = VideoEmbeddingDataset("train")
    val_ds   = VideoEmbeddingDataset("val")

    sample = np.load(train_ds.items[0])
    FEAT_DIM = sample.shape[1]

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=(NUM_WORKERS > 0)
    )

    model = TemporalModel(FEAT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    last_ckpt = CHECKPOINT_DIR / "temporal_last.pth"

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        preds, labels_all = [], []
        loss_sum = 0.0

        for seqs, lengths, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(seqs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += loss.item() * seqs.size(0)
            preds.append(torch.sigmoid(logits).detach().cpu())
            labels_all.append(labels.cpu())

        preds = torch.cat(preds).numpy()
        labels_all = torch.cat(labels_all).numpy()
        train_auc = safe_auc(labels_all, preds)
        train_loss = loss_sum / len(train_ds)

        # Validation
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for seqs, lengths, labels, _ in val_loader:
                seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
                logits, _ = model(seqs, lengths)
                vp.append(torch.sigmoid(logits).cpu())
                vl.append(labels)

        vp = torch.cat(vp).numpy()
        vl = torch.cat(vl).numpy()
        val_auc = safe_auc(vl, vp)
        scheduler.step(val_auc)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_auc": best_val_auc,
            "val_auc": val_auc
        }, last_ckpt)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "temporal_best.pth")

        print(f"Epoch {epoch} | train_auc={train_auc:.4f} val_auc={val_auc:.4f} time={time.time()-t0:.1f}s")

# ---------------- Entry ---------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()