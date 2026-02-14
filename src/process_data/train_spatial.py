# 04 - Train spatial EfficientNet-B3 (frame-level classifier)
# Trains EfficientNet-B3 on frame images (face-cropped). Saves per-epoch checkpoints and best model (best val AUC).
# - Uses mixed precision (autocast + GradScaler)
# - Resumable from checkpoint
# - Deterministic-ish with seed

# Config & imports
import os
from pathlib import Path
import random
import json
from pprint import pprint
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import roc_auc_score

# ---------------- USER CONFIG (edit if you want) ----------------
ROOT = Path.cwd().parent.parent
FRAMES_ROOT = ROOT / "preprocessed" / "frames"   # expects <split>/<video_stem>/frame_*.jpg
LABELS_JSON = ROOT / "data" / "labels.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "spatial"
LOG_DIR = ROOT / "logs"
NUM_EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-4               # starting learning rate
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 50
# ----------------------------------------------------------------

for p in [CHECKPOINT_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Frames root:", FRAMES_ROOT)
print("Labels file:", LABELS_JSON)
print("Checkpoint dir:", CHECKPOINT_DIR)
print(f"Epochs={NUM_EPOCHS} batch_size={BATCH_SIZE} lr={LR} wd={WEIGHT_DECAY}")

# Reproducibility (best-effort)
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Load labels.json (mapping video_stem -> 0/1)
with open(LABELS_JSON, "r") as f:
    labels_map = json.load(f)

# helper to get label from a video folder stem robustly
def get_label_by_stem(stem):
    # direct lookup
    if stem in labels_map:
        return int(labels_map[stem])
    # fallback: if labels keys have suffixes like 'stem__dfdc' try startswith
    candidates = [v for k,v in labels_map.items() if k.startswith(stem)]
    if len(candidates) == 1:
        return int(candidates[0])
    # last fallback: try any key where stem contained
    for k,v in labels_map.items():
        if stem in k:
            return int(v)
    # if not found, raise to catch dataset issues
    raise KeyError(f"Label for stem '{stem}' not found in labels.json")

from PIL import Image

class FrameDataset(Dataset):
    """
    frames_root/<split>/<video_stem>/frame_00.jpg ...
    Each frame is a training sample; label = labels_map[video_stem]
    """
    def __init__(self, split, transform=None):
        self.root = FRAMES_ROOT / split
        self.transform = transform
        # build list of (image_path, label)
        items = []
        if not self.root.exists():
            raise RuntimeError(f"Frames directory not found: {self.root}")
        # iterate video folders
        for video_folder in sorted(self.root.iterdir()):
            if not video_folder.is_dir():
                continue
            stem = video_folder.name
            try:
                label = get_label_by_stem(stem)
            except KeyError:
                # skip if not found (shouldn't happen if labels.json correct)
                continue
            # collect frame images inside folder
            frames = sorted(list(video_folder.glob("frame_*.jpg")))
            for f in frames:
                items.append((str(f), int(label), stem))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, label, stem = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), stem

# transforms (train & val)
train_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# quick debug: create datasets (but don't load full data here)
# ds_train = FrameDataset("train", transform=train_tfms)
# print("Train samples:", len(ds_train))

class SpatialModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b3", pretrained=True, head_hidden=512, dropout=0.4):
        super().__init__()
        # timm model with num_classes=0 returns feature vector
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)  # logits
        )

    def forward(self, x):
        feats = self.backbone(x)   # [B, feat_dim]
        logits = self.head(feats).squeeze(1)
        return logits

# instantiate
model = SpatialModel(pretrained=True).to(DEVICE)

# optimizer, scheduler, loss
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# Compute class imbalance weight (optional but recommended)
pos_count = sum(v for v in labels_map.values())
neg_count = len(labels_map) - pos_count
pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#device_type = "cuda" if torch.cuda.is_available() else "cpu"
scaler = GradScaler()

import time

def save_checkpoint(state, fname):
    # copy model weights to CPU to reduce CUDA memory pressure and make file portable
    cpu_state = state.copy()
    cpu_state["model_state"] = {k: v.cpu() for k, v in state["model_state"].items()}
    # optimizer state may contain tensors — move them to CPU as well (if present)
    if "optimizer_state" in state and state["optimizer_state"] is not None:
        opt_state = state["optimizer_state"]
        # shallow copy
        cpu_opt_state = {}
        cpu_opt_state['state'] = {}
        cpu_opt_state['param_groups'] = opt_state.get('param_groups', [])
        for k, v in opt_state.get('state', {}).items():
            cpu_opt_state['state'][k] = {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv
                                         for sk, sv in v.items()}
        cpu_state["optimizer_state"] = cpu_opt_state
    torch.save(cpu_state, fname)

def compute_auc(y_true, y_scores):
    try:
        return roc_auc_score(y_true, y_scores)
    except Exception:
        return float('nan')


def main():
    # DataLoaders
    train_ds = FrameDataset("train", transform=train_tfms)
    val_ds = FrameDataset("val", transform=val_tfms)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,              
        pin_memory=pin_memory,
        persistent_workers=False    # IMPORTANT: disable
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    best_val_auc = 0.0
    start_epoch = 0

    # Resume logic: load last checkpoint if exists (optional)
    last_ckpt = CHECKPOINT_DIR / "spatial_last.pth"
    if last_ckpt.exists():
        print("Found last checkpoint, resuming:", last_ckpt)
        ck = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        # ensure optimizer state tensors are on the current device
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
        start_epoch = ck.get("epoch", 0) + 1
        best_val_auc = ck.get("best_val_auc", 0.0)
        print("Resumed from epoch", start_epoch, "best_val_auc", best_val_auc)

    # training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", disable=False)
        for i, (imgs, labels, _) in enumerate(pbar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE.type=="cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(logits).detach()
            all_preds.append(preds)
            all_labels.append(labels.detach())

            if (i+1) % PRINT_FREQ == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()
        else:
            all_preds = np.array([])
            all_labels = np.array([])
        
        train_loss = running_loss / len(train_ds)
        train_auc = compute_auc(all_labels, all_preds)

        # validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val {epoch}", unit="batch", disable=False)
            for imgs, labels, _ in vbar:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast(enabled=(DEVICE.type=="cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_preds.append(torch.sigmoid(logits).detach())
                val_labels.append(labels.detach())
                vbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        if len(val_preds) > 0:
            val_preds = torch.cat(val_preds).cpu().numpy()
            val_labels = torch.cat(val_labels).cpu().numpy()
        else:
            val_preds = np.array([])
            val_labels = np.array([])

        val_loss = val_loss / len(val_ds)
        val_auc = compute_auc(val_labels, val_preds)

        # scheduler step (ReduceLROnPlateau)
        scheduler.step(val_auc)

        # save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_auc": best_val_auc,
            "val_auc": val_auc
        }
        last_path = CHECKPOINT_DIR / "spatial_last.pth"
        save_checkpoint(ckpt, last_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_path = CHECKPOINT_DIR / "spatial_best_valAUC.pth"
            save_checkpoint(ckpt, best_path)
            print(f"Saved new best model at epoch {epoch} val_auc={val_auc:.4f}")

        # also save epoch checkpoint (optional)
        epoch_path = CHECKPOINT_DIR / f"spatial_epoch_{epoch}.pth"
        save_checkpoint(ckpt, epoch_path)

        print(f"Epoch {epoch} done. train_loss={train_loss:.4f} train_auc={train_auc:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} time={(time.time()-t0):.1f}s")
        # flush logs if needed

    # quick test evaluation if you want (uses trained best model)
    best_model_path = CHECKPOINT_DIR / "spatial_best_valAUC.pth"
    if best_model_path.exists():
        ck = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        print("Loaded best model with stored best_val_auc:", ck.get("best_val_auc"))
        # compute metrics on test split (frame-level)
        test_ds = FrameDataset("test", transform=val_tfms)
        # use same pin_memory logic as training/validation
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=pin_memory,
                                persistent_workers=(NUM_WORKERS > 0))
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for imgs, labels, _ in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast(enabled=(DEVICE.type == "cuda")):
                    logits = model(imgs)
                test_preds.append(torch.sigmoid(logits).detach())
                test_labels.append(labels.detach())
        # aggregate once
        test_preds = torch.cat(test_preds).cpu().numpy()
        test_labels = torch.cat(test_labels).cpu().numpy()
        test_auc = roc_auc_score(test_labels, test_preds)
        print("Test frame-level AUC:", test_auc)
    else:
        print("No best checkpoint found yet at", best_model_path)

if __name__ == "__main__":
    # On Windows freeze_support ensures safe child process start
    from multiprocessing import freeze_support
    freeze_support()
    main()

