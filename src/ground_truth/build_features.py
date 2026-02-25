# src/ground_truth/build_features.py
import torch
import numpy as np
import json
import sys
from pathlib import Path
from tqdm import tqdm
import traceback

# Import the unified models from local models.py
from models import TemporalModel, SpatialHead

# --- CONFIGURATION ---
ROOT = Path.cwd().parent.parent  # you said files are in src/ground_truth
EMB_DIR = ROOT / "embeddings"
LABELS_PATH = ROOT / "data" / "labels.json"
OUT_DIR = ROOT / "ensemble_features_final"

# Checkpoint Paths
SPATIAL_CKPT = ROOT / "checkpoints" / "spatial" / "spatial_best_valAUC.pth"
TEMPORAL_CKPT = ROOT / "checkpoints" / "temporal" / "temporal_best_valAUC.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _strip_module_prefix(sd: dict):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def _filter_head_state_dict(state_dict: dict):
    """
    From a checkpoint state dict, extract keys that belong to the spatial head.
    This assumes the training saved the head under keys like 'head.<...>'.
    Returns a new dict with 'head.' prefix removed for loading into HeadOnly.
    """
    new = {}
    for k, v in state_dict.items():
        k2 = k.replace("module.", "")
        if k2.startswith("head."):
            new_key = k2[len("head.") :]
            new[new_key] = v
    return new


def load_models():
    print(f"Loading models on {DEVICE}...")

    # --- find a sample embedding to detect feat_dim ---
    sample_file = next(EMB_DIR.rglob("*.npy"), None)
    if not sample_file:
        raise FileNotFoundError("No .npy embedding files found in embeddings/ folder!")

    arr = np.load(sample_file)
    if arr.ndim != 2:
        raise ValueError(f"Sample embedding is not 2-D (T, F). Got shape: {arr.shape}")
    feat_dim = arr.shape[1]
    print(f"Feature dimension detected from data: {feat_dim}")

    # --- Build a head-only spatial module that accepts embedding vectors ---
    spatial_head = SpatialHead(feat_dim=feat_dim).to(DEVICE)

    if SPATIAL_CKPT.exists():
        print(f"Loading Spatial Checkpoint: {SPATIAL_CKPT}")
        ck = torch.load(SPATIAL_CKPT, map_location="cpu")
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        state = _strip_module_prefix(state)
        head_sd = _filter_head_state_dict(state)
        if len(head_sd) == 0:
            print("⚠️ Warning: no 'head.' keys found in spatial checkpoint. Attempting to load full state dict into head (best-effort).")
            # try load by shape matching (best-effort fallback)
            try:
                spatial_head.load_state_dict(state, strict=False)
            except Exception as e:
                print("Failed best-effort head load:", e)
        else:
            try:
                spatial_head.load_state_dict(head_sd, strict=False)
            except Exception as e:
                print("Failed to load head state dict:", e)
        spatial_head.eval()
    else:
        print(f"❌ CRITICAL: Spatial checkpoint not found at {SPATIAL_CKPT}. The spatial head will be randomly initialized.")

    # --- Temporal model (expects sequence of embeddings) ---
    temporal = TemporalModel(feat_dim=feat_dim).to(DEVICE)
    if TEMPORAL_CKPT.exists():
        print(f"Loading Temporal Checkpoint: {TEMPORAL_CKPT}")
        ck = torch.load(TEMPORAL_CKPT, map_location="cpu")
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        state = _strip_module_prefix(state)
        try:
            temporal.load_state_dict(state, strict=False)
        except Exception as e:
            print("Warning: temporal load_state_dict raised:", e)
        temporal.eval()
    else:
        print("⚠️ WARNING: Temporal checkpoint not found. Temporal scores will be from a randomly initialized temporal model!")

    return spatial_head, temporal


def get_label(stem, labels_map):
    if stem in labels_map:
        return int(labels_map[stem])
    for k, v in labels_map.items():
        if stem in k:
            return int(v)
    return None


def process_split(split_name, spatial_head, temporal_model, labels_map):
    split_path = EMB_DIR / split_name
    if not split_path.exists():
        print(f"Skipping {split_name} (folder not found): {split_path}")
        return

    print(f"Processing split: {split_name}...")
    files = sorted(list(split_path.glob("*.npy")))

    X_list = []
    y_list = []
    skipped = 0
    skip_reasons = {}

    for p in tqdm(files):
        try:
            emb = np.load(p)
            # robust checks
            if not isinstance(emb, np.ndarray) or emb.ndim != 2 or emb.shape[0] == 0:
                skipped += 1
                skip_reasons.setdefault("bad_shape", 0)
                skip_reasons["bad_shape"] += 1
                continue
            if np.isnan(emb).any():
                skipped += 1
                skip_reasons.setdefault("nan", 0)
                skip_reasons["nan"] += 1
                continue

            label = get_label(p.stem, labels_map)
            if label is None:
                skipped += 1
                skip_reasons.setdefault("no_label", 0)
                skip_reasons["no_label"] += 1
                continue

            # Convert to tensor
            emb_t = torch.from_numpy(emb).float().to(DEVICE)  # [T, F]

            with torch.no_grad():
                # --- Spatial head: expects [N, feat_dim] where N = T (frames)
                sp_logits = spatial_head(emb_t)  # [T]
                sp_probs = torch.sigmoid(sp_logits).cpu().numpy()
                sp_probs = np.nan_to_num(sp_probs, nan=0.0, posinf=0.0, neginf=0.0)

                s_mean = float(sp_probs.mean())
                s_max = float(sp_probs.max())
                s_std = float(sp_probs.std())
                if len(sp_probs) >= 3:
                    s_top3 = float(np.sort(sp_probs)[-3:].mean())
                else:
                    s_top3 = s_mean

                # --- Temporal feature
                emb_batch = emb_t.unsqueeze(0)  # [1, T, F]
                seq_len = int(emb_t.shape[0])
                lengths = torch.tensor([seq_len], dtype=torch.long, device=DEVICE)

                tm_logit = temporal_model(emb_batch, lengths)  # [1] or scalar tensor
                # make sure tm_logit is tensor and extract float
                if isinstance(tm_logit, torch.Tensor):
                    t_prob = float(torch.sigmoid(tm_logit).cpu().numpy().ravel()[0])
                else:
                    # fallback (unlikely)
                    t_prob = float(torch.sigmoid(torch.tensor(tm_logit)).item())

            features = [s_mean, s_max, s_std, s_top3, t_prob]
            X_list.append(features)
            y_list.append(label)

        except Exception as e:
            skipped += 1
            skip_reasons.setdefault("exception", 0)
            skip_reasons["exception"] += 1
            # optional: debug print for a few examples
            # print(f"Error processing {p.name}: {e}")
            # traceback.print_exc()
            continue

    # Save to disk
    if len(X_list) > 0:
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        out_path = OUT_DIR / f"{split_name}.npz"
        np.savez_compressed(out_path, X=X, y=y)
        print(f"✅ Saved {split_name}: {len(y)} samples (Skipped: {skipped}) -> {out_path}")
    else:
        print(f"❌ No valid samples found for {split_name} (skipped={skipped}), reasons: {skip_reasons}")


if __name__ == "__main__":
    if not LABELS_PATH.exists():
        print(f"Error: Labels file not found at {LABELS_PATH}")
        sys.exit(1)

    with open(LABELS_PATH, "r") as f:
        labels_map = json.load(f)

    try:
        spatial_head, temporal_model = load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Process All Splits (adjust these names to your repo's folders)
    splits = ["train", "val", "test", "test_internal", "reserved_200"]

    for split in splits:
        process_split(split, spatial_head, temporal_model, labels_map)