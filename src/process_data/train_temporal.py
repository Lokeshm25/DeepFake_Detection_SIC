# 06 - Train temporal model (LSTM + attention) on embeddings
# Trains a video-level LSTM aggregator using per-frame embeddings produced by the spatial model.
# Saves checkpoints: checkpoints/temporal/

from pathlib import Path
import json, time
import random
from pprint import pprint
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.cuda.amp import autocast, GradScaler
# AMP disabled for temporal model (FP32 is more stable)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ------------- USER CONFIG -------------
ROOT = Path.cwd().parent.parent  # adjust if running from notebooks/
EMB_ROOT = ROOT / "embeddings"             # embeddings/<split>/<video_stem>.npy
LABELS_JSON = ROOT / "data" / "labels.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "temporal"
NUM_EPOCHS = 25
BATCH_SIZE = 32            # number of videos per batch
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6            # keep 0 in notebooks; increase on robust machines
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 20
LSTM_HIDDEN = 512
LSTM_LAYERS = 2
DROPOUT = 0.3
ATTENTION = True           # use attention pooling over LSTM outputs
BIDIRECTIONAL = True
# ---------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print("Device:", DEVICE)
print("Emb root:", EMB_ROOT)
print("Checkpoint dir:", CHECKPOINT_DIR)
print(f"Epochs={NUM_EPOCHS} batch_size={BATCH_SIZE} lr={LR}")

# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# load labels.json
with open(LABELS_JSON, "r") as f:
    labels_map = json.load(f)

def get_label_from_stem(stem):
    if stem in labels_map:
        return int(labels_map[stem])
    for k,v in labels_map.items():
        if stem in k:
            return int(v)
    raise KeyError(f"Label for {stem} not found")

# Dataset
class VideoEmbeddingDataset(Dataset):
    def __init__(self, split):
        self.root = EMB_ROOT / split
        if not self.root.exists():
            raise RuntimeError(f"No embeddings for split: {split}")
        self.items = sorted([p for p in self.root.glob("*.npy")])
        # optional: filter if empty
        self.items = [p for p in self.items if p.stat().st_size > 0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        stem = p.stem
        arr = np.load(p)                  # shape (T, feat_dim)
        # convert to float32 and torch tensor
        emb = torch.from_numpy(arr.astype(np.float32))  # [T,feat]
        label = get_label_from_stem(stem)
        return emb, torch.tensor(label, dtype=torch.float32), stem

# quick sanity
# ds = VideoEmbeddingDataset("train")
# print("Train videos:", len(ds))

def collate_fn(batch):
    # """
    # batch: list of (emb [T,feat], label, stem)
    # Pads sequences to longest T in batch (simple zero padding).
    # Returns tensors: seqs [B, Tmax, feat], lengths [B], labels [B]
    # """
    seqs, labels, stems = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    feat_dim = seqs[0].shape[1]
    out = torch.zeros(len(seqs), maxlen, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0], :] = s
    labels = torch.stack(labels)
    return out, torch.tensor(lengths, dtype=torch.long), labels, list(stems)

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, h, lengths):
        # h: [B, T, H]
        scores = self.att(h).squeeze(-1)      # [B, T]
        # mask
        mask = torch.arange(h.size(1), device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-1e9"))
        weights = torch.softmax(scores, dim=1)   # [B, T]
        out = (h * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        return out, weights

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
        # """
        # x: [B, T, feat]
        # lengths: [B] (torch.LongTensor)
        # Returns:
        #   logits [B], att_weights [B, T] or None
        # """

        # Run LSTM on padded sequences directly
        # x_out: [B, T, H_out]
        x_out, _ = self.lstm(x)  # no packing; padding positions should be zeros (from collate)
        pooled, att_weights = self.attn(x_out, lengths)
        logits = self.head(pooled).squeeze(1)
        return logits, att_weights

# training loop
def main():
    # Build one dataset to read feat_dim
    train_ds = VideoEmbeddingDataset("train")
    val_ds = VideoEmbeddingDataset("val")
    # sanity check 
    if len(train_ds) == 0:
        raise RuntimeError("No train embeddings found. Run extract_embeddings first.")
    
    sample_emb = np.load(train_ds.items[0])
    FEAT_DIM = int(sample_emb.shape[1])
    print("Feat dim:", FEAT_DIM, "Train videos:", len(train_ds))

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=pin_memory, persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=pin_memory, persistent_workers=(NUM_WORKERS > 0))

    model = TemporalModel(feat_dim=FEAT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    #print(model)

    # quick sanity test
    # feat_dim = FEAT_DIM  # from earlier
    # model_test = TemporalModel(feat_dim=feat_dim).to(DEVICE)
    # B, T = 4, 8
    # dummy = torch.randn(B, T, feat_dim).to(DEVICE)
    # lengths = torch.tensor([8,6,5,7], dtype=torch.long).to(DEVICE)
    # with torch.no_grad():
    #     logits, att = model_test(dummy, lengths)
    # print("logits.shape:", logits.shape, "att shape:", None if att is None else att.shape)
    best_val_auc = 0.0
    start_epoch = 0
    last_ckpt = CHECKPOINT_DIR / "temporal_last.pth"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val_auc = ck.get("best_val_auc", 0.0)
        print("Resumed temporal from", start_epoch, "best", best_val_auc)

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0

        for seqs, lengths, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            seqs = seqs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(seqs, lengths)
            loss = criterion(logits, labels)

            loss.backward()

            # (optional but STRONGLY recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item() * seqs.size(0)
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        train_auc = roc_auc_score(all_labels, all_preds)
        train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for seqs, lengths, labels, stems in tqdm(val_loader):
                seqs = seqs.to(DEVICE); lengths = lengths.to(DEVICE); labels = labels.to(DEVICE)

                logits, _ = model(seqs, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item() * seqs.size(0)
                val_preds.append(torch.sigmoid(logits).cpu())
                val_labels.append(labels.cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_auc = roc_auc_score(val_labels, val_preds)
        val_loss = val_loss / len(val_loader.dataset)

        scheduler.step(val_auc)

        ck = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_auc": best_val_auc,
            "val_auc": val_auc
        }
        torch.save(ck, last_ckpt)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(ck, CHECKPOINT_DIR / "temporal_best.pth")
            print("Saved new best temporal model", best_val_auc)

        print(f"Epoch {epoch} done. train_loss={train_loss:.4f} train_auc={train_auc:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} time={(time.time()-t0):.1f}s")

    # After training: evaluate on test split (video-level)
    best = CHECKPOINT_DIR / "temporal_best.pth"
    if best.exists():
        ck = torch.load(best, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        print("Loaded best temporal model with val_auc:", ck.get("best_val_auc"))
        # test dataset and loader
        test_loader = DataLoader(VideoEmbeddingDataset("test"), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        model.eval()
        t_preds, t_labels, stems_all = [], [], []
        with torch.no_grad():
            for seqs, lengths, labels, stems in test_loader:
                seqs = seqs.to(DEVICE); lengths = lengths.to(DEVICE)
                
                logits, _ = model(seqs, lengths)
                t_preds.append(torch.sigmoid(logits).cpu())
                t_labels.append(labels)
                stems_all.extend(stems)
        t_preds = torch.cat(t_preds).numpy()
        t_labels = torch.cat(t_labels).numpy()
        print("Test AUC:", roc_auc_score(t_labels, t_preds))

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()