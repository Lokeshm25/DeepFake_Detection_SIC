# src/ground_truth/train_ensemble_final.py
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
ROOT = Path.cwd().parent.parent
CACHE_DIR = ROOT / "ensemble_features_final"
CHECKPOINT_DIR = ROOT / "checkpoints" / "ensemble_final"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TXT = CHECKPOINT_DIR / "results.txt"

RANDOM_STATE = 42
NFOLD = 5

def load_data(split_name):
    path = CACHE_DIR / f"{split_name}.npz"
    if not path.exists():
        print(f"⚠️ Warning: {split_name}.npz not found in {CACHE_DIR}")
        return None, None
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if X is None or y is None or len(y) == 0:
        print(f"⚠️ Warning: {split_name} appears empty: {path}")
        return None, None
    print(f"Loaded {split_name}: X={X.shape}, y={y.shape}")
    return X, y

def safe_auc(y_true, probs):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, probs)
    except Exception:
        return float("nan")

def train_and_evaluate():
    print(f"--- Loading Features from {CACHE_DIR} ---")
    X_tr, y_tr = load_data("train")
    X_val, y_val = load_data("val")
    # prefer internal test if available
    X_test, y_test = load_data("test_internal")
    test_name = "Test (Internal)"
    if X_test is None:
        X_test, y_test = load_data("test")
        test_name = "Test (External)"

    if X_tr is None or y_tr is None:
        print("❌ CRITICAL: Train data missing. Run build_features.py first.")
        sys.exit(1)

    # Base pipeline used as "base model" whose OOF predictions become meta-features
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=5000, C=1.0, random_state=RANDOM_STATE))
    ])

    # 1) OOF generation on TRAIN (for stacking)
    print(f"\n--- Generating OOF predictions (n_splits={NFOLD}) ---")
    n = len(y_tr)
    oof_probs = np.zeros((n,), dtype=np.float32)
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=RANDOM_STATE)
    fold_idx = 0
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        fold_idx += 1
        print(f"  Fold {fold_idx}: train {len(tr_idx)} / val {len(va_idx)}")
        X_tr_fold, y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
        X_va_fold = X_tr[va_idx]

        # fit base on fold
        base_pipe.fit(X_tr_fold, y_tr_fold)
        # predict probs on held-out fold
        probs = base_pipe.predict_proba(X_va_fold)[:, 1]
        oof_probs[va_idx] = probs

    # Quick sanity check: any zeros left? (not necessarily an error)
    if np.any(np.isnan(oof_probs)):
        raise RuntimeError("NaNs found in OOF probabilities — check base pipeline / data.")

    train_oof_auc = safe_auc(y_tr, oof_probs)
    print(f"OOF Train AUC: {train_oof_auc:.4f}")

    # 2) Fit meta model on OOF features
    # Here we use a simple logistic regression as meta (input = single column of base-prob)
    meta_X = oof_probs.reshape(-1, 1)
    meta = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)
    meta.fit(meta_X, y_tr)
    print("Meta model trained on OOF predictions.")

    # 3) Fit base on full training set (to produce meta-features for val/test)
    base_full = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=5000, C=1.0, random_state=RANDOM_STATE))
    ])
    base_full.fit(X_tr, y_tr)
    print("Base model retrained on full train set.")

    # Create meta-features for validation and test using the base_full predictions
    val_meta_X = None
    test_meta_X = None
    if X_val is not None:
        val_base_probs = base_full.predict_proba(X_val)[:, 1].reshape(-1, 1)
        val_meta_X = val_base_probs
    if X_test is not None:
        test_base_probs = base_full.predict_proba(X_test)[:, 1].reshape(-1, 1)
        test_meta_X = test_base_probs

    # 4) Calibrate meta on VAL (if available)
    calibrator = None
    if val_meta_X is not None and y_val is not None:
        print("\n--- Calibrating meta model on Validation set ---")
        calibrator = CalibratedClassifierCV(estimator=meta, method="sigmoid", cv="prefit")
        calibrator.fit(val_meta_X, y_val)
        print("Calibrator fitted (sigmoid/Platt).")
    else:
        print("No validation set found — skipping calibration. Using raw meta probabilities.")

    # 5) Evaluate: use OOF for Train, meta (or calibrated meta) for val/test
    results = {}

    # Train (use OOF predictions)
    results["Train (OOF)"] = train_oof_auc

    # Val
    if val_meta_X is not None:
        if calibrator is not None:
            val_probs = calibrator.predict_proba(val_meta_X)[:, 1]
        else:
            val_probs = meta.predict_proba(val_meta_X)[:, 1]
        results["Val"] = safe_auc(y_val, val_probs)
        print(f"Val AUC: {results['Val']:.4f}")
    else:
        print("Val set missing; Val AUC skipped.")

    # Test
    if test_meta_X is not None and y_test is not None:
        if calibrator is not None:
            test_probs = calibrator.predict_proba(test_meta_X)[:, 1]
        else:
            test_probs = meta.predict_proba(test_meta_X)[:, 1]
        results[test_name] = safe_auc(y_test, test_probs)
        print(f"{test_name} AUC: {results[test_name]:.4f}")
    else:
        print(f"{test_name} missing; Test AUC skipped.")

    # Optionally produce classification report on test (threshold 0.5) if present
    if test_meta_X is not None and y_test is not None:
        pred_labels = (test_probs >= 0.5).astype(int)
        report = classification_report(y_test, pred_labels, digits=4)
    else:
        report = "No test report (test missing)."

    # 6) Save artifacts: we save base_full, meta, calibrator (if any)
    artifact = {
        "base_full": base_full,
        "meta": meta,
        "calibrator": calibrator,
        "meta_input_cols": ["base_prob"],
        "notes": "base -> produce base_prob -> meta on oof; calibrate on val if available"
    }
    model_path = CHECKPOINT_DIR / "ensemble_artifacts.joblib"
    joblib.dump(artifact, model_path)

    # 7) Save results
    with open(RESULTS_TXT, "w") as f:
        f.write("Ensemble Results (OOF stacking -> meta -> calibration on VAL)\n")
        f.write("===========================================================\n")
        for k, v in results.items():
            f.write(f"{k} AUC: {np.nan if v is None else v:.4f}\n")
        f.write("\nClassification report (test):\n")
        f.write(report + "\n")

    print(f"\n✅ Saved ensemble artifacts to: {model_path}")
    print(f"✅ Saved results to: {RESULTS_TXT}")
    print("\nDone.")

if __name__ == "__main__":
    train_and_evaluate()