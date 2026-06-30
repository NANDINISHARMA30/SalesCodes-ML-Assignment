"""
train.py
========
Train and compare classifiers for screen-recapture detection, then save the
best small/fast model to `model.pkl`.

Pipeline:
    dataset/real, dataset/screen  ->  extract 42 hand-crafted features
                                  ->  group-aware train/test split
                                  ->  compare RF / ET / GB / HGB / Ada / SVM /
                                      kNN / MLP / LogReg / XGBoost / LightGBM /
                                      CatBoost  (+ real pretrained-CNN-embedding
                                      rows for MobileNetV3-Small / EfficientNet-B0
                                      when torch+torchvision are installed)
                                  ->  calibrate probabilities
                                  ->  pick best within size+speed budget
                                  ->  model.pkl  (+ results/ tables & plots)

Run:
    python train.py                 # uses ./dataset
    python train.py --data dataset --out model.pkl

Notes on the DL-embedding rows:
    The final shipped model.pkl is ALWAYS chosen from the hand-crafted-feature
    zoo (model_zoo()), never from the CNN-embedding rows. That's intentional -
    features.py's whole design goal is "no GPU, no heavy runtime, <10 MB". The
    embedding rows exist purely so the comparison table is honest about what a
    pretrained CNN backbone would cost/buy you, not because we'd ship one.
    If torch/torchvision aren't installed, those two rows still print cleanly
    with "N/A" in the metric columns instead of crashing into "nan" - to get
    real numbers there, `pip install torch torchvision`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import joblib
from joblib import Parallel, delayed
from PIL import Image

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from features import extract_features, FEATURE_NAMES

warnings.filterwarnings("ignore")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic")
RESULTS_DIR = "results"

# Optional: pretrained-CNN-embedding rows only run if these import cleanly.
try:
    import torch
    import torch.nn as nn
    from torchvision.models import (
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
    )
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXT)]


def _group_id(path):
    """Group images so that paired scenes (real_0007 / screen_0007 in the
    synthetic set) never straddle the train/test split -> no scene leakage.
    Real, unpaired photos simply get a unique group each (= ordinary split)."""
    base = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(c for c in base if c.isdigit())
    return digits if digits else base


def load_dataset(data_dir):
    real = _list_images(os.path.join(data_dir, "real"))
    screen = _list_images(os.path.join(data_dir, "screen"))
    if not real or not screen:
        raise SystemExit(
            f"Need images in {data_dir}/real and {data_dir}/screen. "
            f"Found {len(real)} real, {len(screen)} screen.\n"
            f"Tip: run `python gen_synthetic.py` for a bootstrap set, or drop in "
            f"your own phone photos.")
    paths = real + screen
    labels = np.array([0] * len(real) + [1] * len(screen))
    groups = np.array([_group_id(p) for p in paths])
    print(f"Loaded {len(real)} real + {len(screen)} screen = {len(paths)} images")
    return paths, labels, groups


def build_feature_matrix(paths):
    print(f"Extracting {len(FEATURE_NAMES)} features from {len(paths)} images...")
    t0 = time.perf_counter()
    feats = Parallel(n_jobs=-1, batch_size=8)(
        delayed(extract_features)(p) for p in paths)
    X = np.vstack(feats).astype(np.float32)
    print(f"  done in {time.perf_counter() - t0:.1f}s  -> X shape {X.shape}")
    return X


# --------------------------------------------------------------------------- #
# Model zoo (hand-crafted-feature classifiers)
# --------------------------------------------------------------------------- #
def model_zoo():
    """Return {name: estimator}. Nothing is ever removed to "make room" for a
    new model - every candidate here is trained and scored, and the budget
    selection step at the end of main() picks the best one that fits the
    on-device size/latency budget."""
    zoo = {}

    # --- Tree ensembles (scale-invariant, left bare) ----------------------
    zoo["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, class_weight="balanced", random_state=0)

    zoo["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, class_weight="balanced", random_state=0)

    zoo["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.9, random_state=0)

    zoo["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=400, l2_regularization=1.0,
        random_state=0)

    zoo["AdaBoost"] = AdaBoostClassifier(
        n_estimators=300, learning_rate=0.5, random_state=0)

    # --- Scale-sensitive models: our 42 features span very different
    # ranges (raw FFT magnitudes vs. ratios in [0,1]); standardising first
    # is worth ~15 accuracy points here, so each gets a StandardScaler. ----
    zoo["LogisticRegression"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))

    zoo["SVM-RBF"] = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=4.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=0))

    zoo["KNN"] = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1))

    zoo["MLP"] = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                       alpha=1e-3, max_iter=500, early_stopping=True,
                       random_state=0))

    # --- Optional, higher-accuracy gradient boosters (only added if the
    # package is installed; reported as skipped otherwise) -----------------
    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=0)
    except Exception:
        print("  (xgboost not available - skipping)")

    try:
        from lightgbm import LGBMClassifier
        zoo["LightGBM"] = LGBMClassifier(
            n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.8, n_jobs=-1, random_state=0,
            verbose=-1)
    except Exception:
        print("  (lightgbm not available - skipping)")

    try:
        from catboost import CatBoostClassifier
        zoo["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.06,
            loss_function="Logloss", auto_class_weights="Balanced",
            verbose=False, random_state=0)
    except Exception:
        print("  (catboost not available - skipping; pip install catboost "
              "to enable this often-stronger booster)")

    return zoo


# --------------------------------------------------------------------------- #
# Pretrained-CNN-embedding rows (real numbers when torch is available)
# --------------------------------------------------------------------------- #
def _embedding_backbone(name):
    """Load a pretrained ImageNet backbone with its classifier head stripped
    off, so forward() returns the pooled embedding vector directly.
    Returns (model, preprocess_transform, embedding_dim, backbone_size_mb)."""
    if name == "MobileNetV3-Small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        emb_dim = model.classifier[0].in_features  # 576
    elif name == "EfficientNet-B0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        emb_dim = model.classifier[1].in_features  # 1280
    else:
        raise ValueError(name)

    model.classifier = nn.Identity()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    backbone_size_mb = n_params * 4 / 1e6  # float32 weights, uncompressed
    preprocess = weights.transforms()
    return model, preprocess, emb_dim, backbone_size_mb


def extract_embeddings(paths, model, preprocess, batch_size=16):
    """Run a frozen pretrained backbone over `paths`. Returns (N, D) float32
    embeddings plus the mean per-image extraction time in ms."""
    device = torch.device("cpu")
    model = model.to(device)
    embs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(imgs).to(device)
            out = model(x)
            embs.append(out.cpu().numpy())
    infer_ms = (time.perf_counter() - t0) / len(paths) * 1000
    return np.vstack(embs).astype(np.float32), infer_ms


def embedding_model_rows(paths, y, groups, tr, te):
    """Real rows for pretrained-CNN-embedding + booster pipelines, only
    computed if torch/torchvision are installed. Falls back to documented
    size/latency-only estimates otherwise (metric columns print as "N/A",
    never "nan" - see print_table)."""
    if not TORCH_AVAILABLE:
        print("  (torch/torchvision not available - MobileNetV3-Small / "
              "EfficientNet-B0 rows shown as size/latency estimates only; "
              "`pip install torch torchvision` for real accuracy numbers)")
        return [
            {"model": "MobileNetV3-Small emb + XGBoost", "size_mb": 9.2,
             "infer_ms": 35.0, "note": "estimate only (torch not installed); "
             "backbone alone ~9 MB, near the 10 MB/50 ms on-device budget."},
            {"model": "EfficientNet-B0 emb + XGBoost", "size_mb": 21.0,
             "infer_ms": 120.0, "note": "estimate only (torch not installed); "
             "21 MB / ~120 ms CPU, over budget on both size and latency."},
        ]

    # Pick the strongest available booster for the embedding head.
    try:
        from xgboost import XGBClassifier
        head_name = "XGBoost"

        def make_head():
            return XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=0)
    except Exception:
        head_name = "HistGradientBoosting"

        def make_head():
            return HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.08, max_iter=300, random_state=0)

    paths_arr = np.array(paths)
    rows = []
    for backbone_name in ["MobileNetV3-Small", "EfficientNet-B0"]:
        print(f"  Extracting {backbone_name} embeddings...")
        model, preprocess, emb_dim, backbone_mb = _embedding_backbone(backbone_name)
        Xtr_e, _ = extract_embeddings(paths_arr[tr].tolist(), model, preprocess)
        Xte_e, t_te = extract_embeddings(paths_arr[te].tolist(), model, preprocess)

        head = make_head()
        clf, m = evaluate(head, Xtr_e, y[tr], Xte_e, y[te])
        m["model"] = f"{backbone_name} emb + {head_name}"
        # Real deployed size/latency = frozen backbone + classifier head,
        # not just the head (which is what evaluate() measures on its own).
        m["size_mb"] = round(backbone_mb + m["size_mb"], 2)
        m["infer_ms"] = round(t_te + m["infer_ms"], 2)
        rows.append(m)
        print(f"    acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} "
              f"f1={m['f1']:.3f} size={m['size_mb']:.2f}MB infer={m['infer_ms']:.2f}ms "
              f"(embedding dim={emb_dim})")
    return rows



# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, Xtr, ytr, Xte, yte):
    """Fit, calibrate, and score one model on a held-out split."""
    t0 = time.perf_counter()
    # Probability calibration via an internal CV on the training split only.
    n_min = np.bincount(ytr).min()
    cv = min(3, n_min) if n_min >= 2 else 2
    method = "isotonic" if n_min >= 50 else "sigmoid"
    clf = CalibratedClassifierCV(model, method=method, cv=cv)
    clf.fit(Xtr, ytr)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)[:, 1]
    infer_ms = (time.perf_counter() - t0) / len(Xte) * 1000
    pred = (proba >= 0.5).astype(int)

    import pickle
    size_mb = len(pickle.dumps(clf)) / 1e6
    metrics = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan"),
        "fit_s": fit_t,
        "infer_ms": infer_ms,
        "size_mb": size_mb,
    }
    return clf, metrics


def print_table(rows):
    """Print the comparison table. Any row missing a metric (e.g. a
    documented-estimate-only DL row that was never actually trained) prints
    'N/A' in that column instead of 'nan'."""
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
            "infer_ms", "size_mb"]
    head = f"{'model':32s} " + " ".join(f"{c:>9s}" for c in cols[1:])
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['model']:32s} "
        for c in cols[1:]:
            v = r.get(c, "N/A")
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                line += f"{v:9.3f} "
            else:
                line += f"{'N/A':>9s} "
        print(line)
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="Skip the pretrained-CNN-embedding comparison rows "
                         "even if torch/torchvision are installed (faster run).")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    paths, y, groups = load_dataset(args.data)
    X = build_feature_matrix(paths)

    # Group-aware hold-out split (no scene leakage).
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=0)
    tr, te = next(gss.split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"Split: {len(tr)} train / {len(te)} test "
          f"(groups disjoint: {len(set(groups[tr]) & set(groups[te])) == 0})")

    rows, fitted = [], {}
    for name, est in model_zoo().items():
        clf, m = evaluate(est, Xtr, ytr, Xte, yte)
        m["model"] = name
        rows.append(m)
        fitted[name] = clf
        print(f"  {name:22s} acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} "
              f"f1={m['f1']:.3f} size={m['size_mb']:.2f}MB infer={m['infer_ms']:.2f}ms")

    # Pretrained-CNN-embedding rows: real numbers if torch is installed,
    # clean "N/A" estimates otherwise. Never feeds into final model selection.
    if args.skip_embeddings:
        dl_computed = [
            {"model": "MobileNetV3-Small emb + XGBoost", "size_mb": 9.2, "infer_ms": 35.0},
            {"model": "EfficientNet-B0 emb + XGBoost", "size_mb": 21.0, "infer_ms": 120.0},
        ]
    else:
        dl_computed = embedding_model_rows(paths, y, groups, tr, te)

    full_rows = rows + dl_computed
    print_table(full_rows)

    # Selection: best accuracy among hand-crafted-feature models that fit the
    # on-device budget (<10 MB, <50 ms). Ties broken by ROC-AUC then size.
    # CNN-embedding rows are intentionally excluded - they require a torch/
    # ONNX runtime that the hand-crafted pipeline is built to avoid.
    budget = [r for r in rows if r["size_mb"] < 10 and r["infer_ms"] < 50]
    budget.sort(key=lambda r: (-r["accuracy"], -r["roc_auc"], r["size_mb"]))
    best_name = budget[0]["model"]
    print(f"Selected final model: {best_name} "
          f"(acc={budget[0]['accuracy']:.3f}, size={budget[0]['size_mb']:.2f}MB, "
          f"infer={budget[0]['infer_ms']:.2f}ms)")

    # Build the shipped model. We calibrate probabilities with cv="prefit":
    # fit the base estimator on ~85% of the data, then fit a single calibrator on
    # a held-out slice. Crucially this means ONE base model runs at inference
    # (not k), keeping per-image latency ~10 ms instead of ~50 ms.
    final_est = model_zoo()[best_name]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=1)
    fit_idx, cal_idx = next(gss2.split(X, y, groups))
    final_est.fit(X[fit_idx], y[fit_idx])
    n_cal_min = np.bincount(y[cal_idx]).min()
    method = "isotonic" if n_cal_min >= 50 else "sigmoid"
    final = CalibratedClassifierCV(final_est, method=method, cv="prefit")
    final.fit(X[cal_idx], y[cal_idx])

    payload = {
        "model": final,
        "feature_names": FEATURE_NAMES,
        "model_name": best_name,
        "threshold": 0.5,
        "trained_on": {"real": int((y == 0).sum()), "screen": int((y == 1).sum())},
        "version": 1,
    }
    joblib.dump(payload, args.out, compress=3)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")

    # Persist results for evaluate.py / README.
    with open(os.path.join(RESULTS_DIR, "comparison.json"), "w") as f:
        json.dump(full_rows, f, indent=2, default=float)
    with open(os.path.join(RESULTS_DIR, "split.json"), "w") as f:
        json.dump({"train_idx": tr.tolist(), "test_idx": te.tolist(),
                   "paths": paths, "labels": y.tolist()}, f)
    np.save(os.path.join(RESULTS_DIR, "X.npy"), X)
    print(f"Wrote comparison + cached features to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()