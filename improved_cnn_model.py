# ============================================================
# ICCAD-12 CROSS-BENCHMARK GENERALISATION  (REVISED VERSION)
#
# What this script does
# ---------------------
# 1. Trains 5 INDEPENDENT models (one per benchmark, never merged)
#    and tests each on all 5 benchmarks  -> 5 x 5 matrices.
# 2. Repeats everything over multiple SEEDS (mean +/- std).
# 3. ABLATION: architecture / loss / augmentation variants,
#    all evaluated with the same 5 x 5 protocol.
# 4. Separates ARCHITECTURE effects (diagonal = in-benchmark)
#    from GENERALISATION (off-diagonal) via the generalisation gap.
# 5. DATA-DRIVEN TRANSFER ANALYSIS:
#    class balance, pattern statistics (density, edge density),
#    raw-pixel MMD, embedding MMD, and their Spearman correlation
#    with cross-benchmark performance.
#
# Resumable: every (variant, seed, source) run is saved to disk
# and skipped if it already exists.
#
# Requirements: tensorflow>=2.11, numpy, pandas, scikit-learn,
#               scipy, matplotlib
# ============================================================

import os
import glob
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")                      # no blocking plt.show()
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    average_precision_score, classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

# CHANGE THIS PATH
DATASET_PATH = r"C:\Users\Shrihita\Documents\aiml proj\iccad_official\iccad-official"

BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

# Make sure the PAPER says the same resolution as this value.
IMAGE_SIZE = (64, 64)

BATCH_SIZE = 32
EPOCHS = 60
VALIDATION_SPLIT = 0.20

# Seeds: main model gets more seeds, ablation variants fewer (saves time)
SEEDS_FULL = [42, 43, 44, 45, 46]
SEEDS_ABLATION = [42, 43, 44]

# ---- Ablation variants -------------------------------------
# loss: "focal" or "bce"; use_class_weight: apply class_weight in fit
VARIANTS = {
    "full":        dict(use_residual=True,  use_se=True,  use_aug=True,  loss="focal", use_class_weight=True),
    "no_se":       dict(use_residual=True,  use_se=False, use_aug=True,  loss="focal", use_class_weight=True),
    "no_residual": dict(use_residual=False, use_se=True,  use_aug=True,  loss="focal", use_class_weight=True),
    "plain_cnn":   dict(use_residual=False, use_se=False, use_aug=True,  loss="focal", use_class_weight=True),
    "no_aug":      dict(use_residual=True,  use_se=True,  use_aug=False, loss="focal", use_class_weight=True),
    # imbalance-handling ablation (focal + class weights double-counts imbalance)
    "focal_only":  dict(use_residual=True,  use_se=True,  use_aug=True,  loss="focal", use_class_weight=False),
    "bce_cw":      dict(use_residual=True,  use_se=True,  use_aug=True,  loss="bce",   use_class_weight=True),
}

# ---- Phases -------------------------------------------------
RUN_TRAINING = True
RUN_AGGREGATION = True
RUN_TRANSFER_ANALYSIS = True

# Quick pipeline test: 2 epochs, 1 seed, only the full variant
QUICK_TEST = False
if QUICK_TEST:
    EPOCHS = 2
    SEEDS_FULL = [42]
    SEEDS_ABLATION = [42]
    VARIANTS = {"full": VARIANTS["full"]}

# MMD settings
MMD_SUBSAMPLE = 1000        # points per set (MMD cost grows quadratically)

# ---- Folders ------------------------------------------------
MODEL_DIR = os.path.join(DATASET_PATH, "models")
RESULT_DIR = os.path.join(DATASET_PATH, "results")
RUN_DIR = os.path.join(RESULT_DIR, "runs")
PROB_DIR = os.path.join(RESULT_DIR, "probabilities")
REPORT_DIR = os.path.join(RESULT_DIR, "classification_reports")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
ANALYSIS_DIR = os.path.join(RESULT_DIR, "transfer_analysis")

for d in [MODEL_DIR, RESULT_DIR, RUN_DIR, PROB_DIR, REPORT_DIR, FIG_DIR, ANALYSIS_DIR]:
    os.makedirs(d, exist_ok=True)

METRICS = [
    "Accuracy", "Precision", "Recall", "F1", "Sensitivity",
    "Specificity", "Balanced Accuracy", "MCC", "ROC-AUC", "PR-AUC"
]
KEY_METRICS = ["Balanced Accuracy", "F1", "MCC", "ROC-AUC", "PR-AUC"]


# ============================================================
# 2. DATA LOADING (benchmarks are NEVER merged)
# ============================================================

CLASS_MAPPING = {
    "hotspot": 1, "hotspots": 1, "hs": 1, "train_hs": 1, "test_hs": 1,
    "non_hotspot": 0, "non-hotspot": 0, "nonhotspot": 0,
    "nhs": 0, "train_nhs": 0, "test_nhs": 0,
}


def load_images(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    images, labels = [], []

    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue

        class_key = class_name.lower().strip()
        if class_key not in CLASS_MAPPING:
            print(f"Skipping unknown folder: {class_name}")
            continue

        label = CLASS_MAPPING[class_key]
        print(f"Loading {class_name}: class {label}")

        for filename in sorted(os.listdir(class_path)):
            filepath = os.path.join(class_path, filename)
            try:
                img = tf.keras.utils.load_img(
                    filepath, target_size=IMAGE_SIZE, color_mode="grayscale"
                )
                img = tf.keras.utils.img_to_array(img) / 255.0
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Could not load {filepath}: {e}")

    if len(images) == 0:
        raise ValueError(f"No images found in {directory}")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_benchmark(benchmark):
    print("\n" + "=" * 70)
    print("LOADING:", benchmark)
    print("=" * 70)

    X_train, y_train = load_images(os.path.join(DATASET_PATH, benchmark, "train"))
    X_test, y_test = load_images(os.path.join(DATASET_PATH, benchmark, "test"))

    print("Train:", len(X_train), "| hotspots:", int(np.sum(y_train == 1)),
          "| non-hotspots:", int(np.sum(y_train == 0)))
    print("Test :", len(X_test), "| hotspots:", int(np.sum(y_test == 1)),
          "| non-hotspots:", int(np.sum(y_test == 0)))

    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test}


def load_all_data():
    return {b: load_benchmark(b) for b in BENCHMARKS}


# ============================================================
# 3. MODEL (with ablation switches)
# ============================================================

def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal_and_vertical"),
            layers.RandomRotation(factor=0.05),
            layers.RandomZoom(height_factor=0.05, width_factor=0.05),
        ],
        name="data_augmentation",
    )


def residual_block(x, filters, dropout_rate, use_residual=True):
    shortcut = x

    y = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    if use_residual:
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.Add()([y, shortcut])

    y = layers.ReLU()(y)
    y = layers.Dropout(dropout_rate)(y)
    return y


def se_attention(x, reduction=8):
    channels = x.shape[-1]
    a = layers.GlobalAveragePooling2D()(x)
    a = layers.Dense(max(channels // reduction, 4), activation="relu")(a)
    a = layers.Dense(channels, activation="sigmoid")(a)
    a = layers.Reshape((1, 1, channels))(a)
    return layers.Multiply()([x, a])


def build_model(use_residual=True, use_se=True, use_aug=True):
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    x = build_augmentation()(inputs) if use_aug else inputs

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 32, 0.05, use_residual)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 64, 0.10, use_residual)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128, 0.15, use_residual)

    if use_se:
        x = se_attention(x)

    x = layers.GlobalAveragePooling2D()(x)

    # This layer is used as the EMBEDDING for the MMD analysis
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="embedding")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="Hotspot_CNN")


def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        pos = -alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
        neg = -(1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(pos + neg)
    return loss


def get_class_weights(y):
    neg, pos, total = np.sum(y == 0), np.sum(y == 1), len(y)
    return {0: total / (2.0 * neg), 1: total / (2.0 * pos)}


# ============================================================
# 4. THRESHOLD + METRICS
# (threshold chosen on SOURCE VALIDATION data only)
# ============================================================

def balanced_accuracy_from_preds(y_true, preds):
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sens + spec) / 2.0


def find_best_threshold(y_true, probabilities):
    best_t, best_s = 0.50, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        s = balanced_accuracy_from_preds(y_true, (probabilities >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


def calculate_metrics(y_true, probabilities, threshold):
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc = np.nan
    try:
        pr = average_precision_score(y_true, probabilities)
    except ValueError:
        pr = np.nan

    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Sensitivity": sens,
        "Specificity": spec,
        "Balanced Accuracy": (sens + spec) / 2.0,
        "MCC": matthews_corrcoef(y_true, preds),
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Threshold": threshold,
        # PR-AUC baseline = prevalence; differs across benchmarks
        "Test Prevalence": float(np.mean(y_true)),
    }


# ============================================================
# 5. PLOTTING HELPERS
# ============================================================

def save_confusion_matrix(cm, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Non-Hotspot", "Hotspot"])
    plt.yticks([0, 1], ["Non-Hotspot", "Hotspot"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_heatmap(mean_mat, std_mat, title, cbar_label, path):
    plt.figure(figsize=(8.5, 7))
    plt.imshow(mean_mat.values, aspect="auto")
    plt.colorbar(label=cbar_label)
    plt.xticks(range(len(BENCHMARKS)), BENCHMARKS)
    plt.yticks(range(len(BENCHMARKS)), BENCHMARKS)
    plt.xlabel("Test Benchmark")
    plt.ylabel("Training Benchmark")
    plt.title(title)
    for i in range(len(BENCHMARKS)):
        for j in range(len(BENCHMARKS)):
            m = mean_mat.iloc[i, j]
            s = std_mat.iloc[i, j] if std_mat is not None else np.nan
            txt = f"{m:.3f}" if np.isnan(s) else f"{m:.3f}\n±{s:.3f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# 6. ONE RUN = (variant, seed, source benchmark)
#    -> trains one model, tests it on all 5 benchmarks
# ============================================================

def run_one(variant, seed, source, all_data):
    run_name = f"{variant}_seed{seed}_{source}"
    run_csv = os.path.join(RUN_DIR, f"{run_name}.csv")

    if os.path.exists(run_csv):
        print(f"[skip] {run_name} already done")
        return

    cfg = VARIANTS[variant]

    print("\n" + "#" * 90)
    print(f"RUN: variant={variant} | seed={seed} | train benchmark={source}")
    print("#" * 90)

    # Seeds python, numpy and TensorFlow (including augmentation layers)
    tf.keras.utils.set_random_seed(seed)

    X = all_data[source]["X_train"]
    y = all_data[source]["y_train"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=seed
    )

    model = build_model(cfg["use_residual"], cfg["use_se"], cfg["use_aug"])

    loss = focal_loss(2.0, 0.75) if cfg["loss"] == "focal" else "binary_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=10, mode="max",
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4,
                          mode="max", min_lr=1e-6),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=get_class_weights(y_tr) if cfg["use_class_weight"] else None,
        callbacks=callbacks,
        verbose=2,
    )

    # Threshold from SOURCE validation only
    val_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    threshold, val_ba = find_best_threshold(y_val, val_probs)
    print(f"Threshold={threshold:.2f} | validation BA={val_ba:.4f}")

    # Save model (restore_best_weights already applied)
    model.save(os.path.join(MODEL_DIR, f"{run_name}.keras"))

    is_main_run = (variant == "full" and seed == SEEDS_FULL[0])

    if is_main_run:
        plt.figure(figsize=(7, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{source} - Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{source}_training_loss.png"), dpi=300)
        plt.close()

    rows, probs_store = [], {}

    for target in BENCHMARKS:
        X_te = all_data[target]["X_test"]
        y_te = all_data[target]["y_test"]

        probs = model.predict(X_te, batch_size=BATCH_SIZE, verbose=0).ravel()
        probs_store[target] = probs

        m = calculate_metrics(y_te, probs, threshold)
        m.update({
            "Variant": variant,
            "Seed": seed,
            "Train Benchmark": source,
            "Test Benchmark": target,
            "Val Threshold": threshold,
            "Val BA": val_ba,
            "Epochs Trained": len(history.history["loss"]),
        })
        rows.append(m)

        print(f"  {source} -> {target}: BA={m['Balanced Accuracy']:.4f} "
              f"F1={m['F1']:.4f} AUC={m['ROC-AUC']:.4f}")

        # Classification reports + confusion matrices for the main model
        if variant == "full":
            preds = (probs >= threshold).astype(int)
            report = classification_report(
                y_te, preds, target_names=["Non-Hotspot", "Hotspot"],
                digits=4, zero_division=0
            )
            with open(os.path.join(
                    REPORT_DIR, f"{run_name}_TO_{target}.txt"), "w") as f:
                f.write(f"Train: {source} | Test: {target} | seed={seed} "
                        f"| threshold={threshold:.2f}\n\n{report}")

            if is_main_run:
                cm = confusion_matrix(y_te, preds, labels=[0, 1])
                save_confusion_matrix(
                    cm, f"Train {source} → Test {target}",
                    os.path.join(FIG_DIR, f"{source}_TO_{target}_confusion_matrix.png"))

    np.savez_compressed(os.path.join(PROB_DIR, f"{run_name}.npz"), **probs_store)

    # Save LAST so that an interrupted run is redone
    pd.DataFrame(rows).to_csv(run_csv, index=False)

    tf.keras.backend.clear_session()


def run_all_training(all_data):
    for variant in VARIANTS:
        seeds = SEEDS_FULL if variant == "full" else SEEDS_ABLATION
        for seed in seeds:
            for source in BENCHMARKS:
                run_one(variant, seed, source, all_data)


# ============================================================
# 7. AGGREGATION (mean +/- std, ablation, generalisation gap)
# ============================================================

def load_all_runs():
    files = sorted(glob.glob(os.path.join(RUN_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError("No run files found. Run training first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(RESULT_DIR, "ALL_RESULTS_ALL_SEEDS_ALL_VARIANTS.csv"),
              index=False)
    return df


def ms(series):
    """mean ± std string"""
    return f"{series.mean():.4f} ± {series.std(ddof=1) if len(series) > 1 else 0.0:.4f}"


def aggregate_main_model(results_df):
    full = results_df[results_df["Variant"] == "full"].copy()
    n_seeds = full["Seed"].nunique()
    print(f"\nMain model: {n_seeds} seeds -> {sorted(full['Seed'].unique())}")

    # --- 5x5 mean / std matrices -------------------------------
    for metric in METRICS:
        mean_m = full.pivot_table(index="Train Benchmark", columns="Test Benchmark",
                                  values=metric, aggfunc="mean"
                                  ).reindex(index=BENCHMARKS, columns=BENCHMARKS)
        std_m = full.pivot_table(index="Train Benchmark", columns="Test Benchmark",
                                 values=metric, aggfunc="std"
                                 ).reindex(index=BENCHMARKS, columns=BENCHMARKS)

        safe = metric.replace(" ", "_")
        mean_m.to_csv(os.path.join(RESULT_DIR, f"{safe}_5x5_MEAN.csv"))
        std_m.to_csv(os.path.join(RESULT_DIR, f"{safe}_5x5_STD.csv"))

        str_m = mean_m.copy().astype(object)
        for i in BENCHMARKS:
            for j in BENCHMARKS:
                str_m.loc[i, j] = f"{mean_m.loc[i, j]:.3f} ± {std_m.loc[i, j]:.3f}"
        str_m.to_csv(os.path.join(RESULT_DIR, f"{safe}_5x5_MEAN_PM_STD.csv"))

        print("\n" + "=" * 90)
        print(f"{metric.upper()} - 5x5 (mean ± std over {n_seeds} seeds)")
        print("=" * 90)
        print(str_m.to_string())

        if metric in ("Balanced Accuracy", "F1", "MCC", "ROC-AUC"):
            save_heatmap(mean_m, std_m,
                         f"ICCAD-12 cross-benchmark {metric} (mean ± std)",
                         metric,
                         os.path.join(FIG_DIR, f"{safe}_5x5_heatmap.png"))

    # --- source / target summaries (off-diagonal) --------------
    off = full[full["Train Benchmark"] != full["Test Benchmark"]]

    src = off.groupby(["Seed", "Train Benchmark"])[METRICS].mean().reset_index()
    src_summary = src.groupby("Train Benchmark")[METRICS].agg(["mean", "std"])
    src_summary.to_csv(os.path.join(RESULT_DIR, "SOURCE_TRANSFER_SUMMARY.csv"))

    tgt = off.groupby(["Seed", "Test Benchmark"])[METRICS].mean().reset_index()
    tgt_summary = tgt.groupby("Test Benchmark")[METRICS].agg(["mean", "std"])
    tgt_summary.to_csv(os.path.join(RESULT_DIR, "TARGET_GENERALIZATION_SUMMARY.csv"))

    # --- pair ranking ------------------------------------------
    pair = (off.groupby(["Train Benchmark", "Test Benchmark"])[
        ["Balanced Accuracy", "F1", "Precision", "Recall", "MCC", "ROC-AUC", "PR-AUC"]]
        .mean().reset_index().sort_values("Balanced Accuracy", ascending=False))
    pair.to_csv(os.path.join(RESULT_DIR, "TRANSFER_PAIR_ANALYSIS.csv"), index=False)
    print("\nTrain -> Test pairs ordered by mean Balanced Accuracy")
    print(pair.round(4).to_string(index=False))


def aggregate_ablation_and_gap(results_df):
    """
    ARCHITECTURE vs GENERALISATION
      diagonal      = in-benchmark performance      -> architecture / fit quality
      off-diagonal  = cross-benchmark performance   -> generalisation
      gap           = diagonal - off-diagonal       -> generalisation loss
    Computed per (variant, seed) first, then mean ± std across seeds.
    """
    rows = []
    for (variant, seed), g in results_df.groupby(["Variant", "Seed"]):
        diag = g[g["Train Benchmark"] == g["Test Benchmark"]]
        off = g[g["Train Benchmark"] != g["Test Benchmark"]]
        for metric in KEY_METRICS:
            rows.append({
                "Variant": variant, "Seed": seed, "Metric": metric,
                "Diagonal": diag[metric].mean(),
                "Off-diagonal": off[metric].mean(),
                "Gap": diag[metric].mean() - off[metric].mean(),
            })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(os.path.join(RESULT_DIR, "GAP_PER_SEED.csv"), index=False)

    out = []
    for (variant, metric), g in long_df.groupby(["Variant", "Metric"]):
        out.append({
            "Variant": variant, "Metric": metric, "Seeds": len(g),
            "Diagonal (in-benchmark)": ms(g["Diagonal"]),
            "Off-diagonal (cross-benchmark)": ms(g["Off-diagonal"]),
            "Gap (diag - off)": ms(g["Gap"]),
        })
    summary = pd.DataFrame(out)

    # Order: full first, then the rest
    order = ["full"] + [v for v in VARIANTS if v != "full"]
    summary["_o"] = summary["Variant"].map({v: i for i, v in enumerate(order)})
    summary = summary.sort_values(["Metric", "_o"]).drop(columns="_o")
    summary.to_csv(os.path.join(RESULT_DIR, "ABLATION_AND_GENERALISATION_GAP.csv"),
                   index=False)

    print("\n" + "=" * 100)
    print("ABLATION + GENERALISATION GAP (mean ± std over seeds)")
    print("=" * 100)
    for metric in KEY_METRICS:
        print(f"\n--- {metric} ---")
        print(summary[summary["Metric"] == metric].drop(columns="Metric")
              .to_string(index=False))

    # Per-variant 5x5 balanced accuracy means (for supplementary tables)
    for variant in results_df["Variant"].unique():
        v = results_df[results_df["Variant"] == variant]
        mat = v.pivot_table(index="Train Benchmark", columns="Test Benchmark",
                            values="Balanced Accuracy", aggfunc="mean"
                            ).reindex(index=BENCHMARKS, columns=BENCHMARKS)
        mat.to_csv(os.path.join(RESULT_DIR, f"ABLATION_{variant}_BA_5x5_MEAN.csv"))


# ============================================================
# 8. DATA-DRIVEN TRANSFER ANALYSIS
# ============================================================

def edge_density(X):
    """Mean absolute horizontal+vertical gradient per image."""
    gx = np.abs(np.diff(X, axis=2)).mean(axis=(1, 2, 3))
    gy = np.abs(np.diff(X, axis=1)).mean(axis=(1, 2, 3))
    return gx + gy


def pixel_density(X):
    return X.reshape(len(X), -1).mean(axis=1)


def compute_dataset_statistics(all_data):
    """Class balance + pattern statistics for every benchmark/split/class."""
    balance_rows, pattern_rows = [], []

    for b in BENCHMARKS:
        for split in ["train", "test"]:
            X = all_data[b][f"X_{split}"]
            y = all_data[b][f"y_{split}"]

            n_hs, n_nhs = int(np.sum(y == 1)), int(np.sum(y == 0))
            balance_rows.append({
                "Benchmark": b, "Split": split, "Images": len(y),
                "Hotspots": n_hs, "Non-Hotspots": n_nhs,
                "Hotspot Ratio": n_hs / len(y),
                "Imbalance (NHS:HS)": (n_nhs / n_hs) if n_hs > 0 else np.nan,
            })

            for name, mask in [("all", np.ones(len(y), bool)),
                               ("hotspot", y == 1),
                               ("non-hotspot", y == 0)]:
                Xc = X[mask]
                if len(Xc) == 0:
                    continue
                d, e = pixel_density(Xc), edge_density(Xc)
                pattern_rows.append({
                    "Benchmark": b, "Split": split, "Class": name, "N": len(Xc),
                    "Density Mean": d.mean(), "Density Std": d.std(),
                    "Edge Density Mean": e.mean(), "Edge Density Std": e.std(),
                })

    bal = pd.DataFrame(balance_rows)
    pat = pd.DataFrame(pattern_rows)
    bal.to_csv(os.path.join(ANALYSIS_DIR, "CLASS_BALANCE.csv"), index=False)
    pat.to_csv(os.path.join(ANALYSIS_DIR, "PATTERN_STATISTICS.csv"), index=False)

    print("\n" + "=" * 90)
    print("CLASS BALANCE")
    print("=" * 90)
    print(bal.round(4).to_string(index=False))
    print("\nPATTERN STATISTICS")
    print(pat.round(4).to_string(index=False))
    return bal, pat


def median_gamma(Z):
    d2 = euclidean_distances(Z, squared=True)
    vals = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(vals)
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def mmd2_unbiased(X, Y, gamma):
    n, m = len(X), len(Y)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    term_x = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(term_x + term_y - 2.0 * Kxy.mean())


def mmd_between(A, B, rng):
    """RBF-MMD^2 with median-heuristic bandwidth on subsampled sets."""
    ia = rng.choice(len(A), min(len(A), MMD_SUBSAMPLE), replace=False)
    ib = rng.choice(len(B), min(len(B), MMD_SUBSAMPLE), replace=False)
    A, B = A[ia], B[ib]
    gamma = median_gamma(np.vstack([A, B]))
    return mmd2_unbiased(A, B, gamma)


def transfer_analysis(all_data, results_df):
    bal, pat = compute_dataset_statistics(all_data)

    # Lookups for pair-level dataset differences
    def stat(b, split, col):
        r = pat[(pat.Benchmark == b) & (pat.Split == split) & (pat.Class == "all")]
        return float(r[col].iloc[0])

    def ratio(b, split):
        r = bal[(bal.Benchmark == b) & (bal.Split == split)]
        return float(r["Hotspot Ratio"].iloc[0])

    # ---- model-independent raw-pixel MMD ----------------------
    print("\nComputing raw-pixel MMD ...")
    rng = np.random.RandomState(0)
    flat = {b: {s: all_data[b][f"X_{s}"].reshape(len(all_data[b][f"X_{s}"]), -1)
                for s in ["train", "test"]} for b in BENCHMARKS}

    pixel_mmd = {}
    for s in BENCHMARKS:
        for t in BENCHMARKS:
            pixel_mmd[(s, t)] = mmd_between(flat[s]["train"], flat[t]["test"], rng)

    # ---- embedding MMD from trained 'full' models -------------
    print("Computing embedding MMD from trained models ...")
    emb_rows = []
    for seed in SEEDS_FULL:
        for s in BENCHMARKS:
            model_path = os.path.join(MODEL_DIR, f"full_seed{seed}_{s}.keras")
            if not os.path.exists(model_path):
                continue
            model = tf.keras.models.load_model(model_path, compile=False)
            extractor = Model(model.input, model.get_layer("embedding").output)

            src_emb = extractor.predict(all_data[s]["X_train"],
                                        batch_size=BATCH_SIZE, verbose=0)
            for t in BENCHMARKS:
                tgt_emb = extractor.predict(all_data[t]["X_test"],
                                            batch_size=BATCH_SIZE, verbose=0)
                emb_rows.append({
                    "Seed": seed, "Train Benchmark": s, "Test Benchmark": t,
                    "Embedding MMD": mmd_between(src_emb, tgt_emb,
                                                 np.random.RandomState(seed)),
                })
            tf.keras.backend.clear_session()

    emb_df = pd.DataFrame(emb_rows)
    emb_df.to_csv(os.path.join(ANALYSIS_DIR, "EMBEDDING_MMD_PER_SEED.csv"), index=False)
    emb_mean = emb_df.groupby(["Train Benchmark", "Test Benchmark"])["Embedding MMD"].mean()

    # ---- performance (full variant, mean over seeds) ----------
    full = results_df[results_df["Variant"] == "full"]
    perf = full.groupby(["Train Benchmark", "Test Benchmark"])[
        ["Balanced Accuracy", "F1", "MCC", "ROC-AUC", "PR-AUC"]].mean()

    # ---- pair table -------------------------------------------
    rows = []
    for s in BENCHMARKS:
        for t in BENCHMARKS:
            rows.append({
                "Train Benchmark": s, "Test Benchmark": t,
                "Is Diagonal": s == t,
                "Embedding MMD": emb_mean.get((s, t), np.nan),
                "Pixel MMD": pixel_mmd[(s, t)],
                "Hotspot Ratio Diff": abs(ratio(s, "train") - ratio(t, "test")),
                "Density Diff": abs(stat(s, "train", "Density Mean") - stat(t, "test", "Density Mean")),
                "Edge Density Diff": abs(stat(s, "train", "Edge Density Mean") - stat(t, "test", "Edge Density Mean")),
                **perf.loc[(s, t)].to_dict(),
            })
    pairs = pd.DataFrame(rows)
    pairs.to_csv(os.path.join(ANALYSIS_DIR, "PAIR_DISTANCE_AND_PERFORMANCE.csv"), index=False)

    # ---- Spearman correlations (20 off-diagonal pairs) --------
    off = pairs[~pairs["Is Diagonal"]]
    features = ["Embedding MMD", "Pixel MMD", "Hotspot Ratio Diff",
                "Density Diff", "Edge Density Diff"]
    outcomes = ["Balanced Accuracy", "F1", "MCC", "ROC-AUC"]

    corr_rows = []
    for f in features:
        for o in outcomes:
            sub = off[[f, o]].dropna()
            if len(sub) < 4:
                continue
            rho, p = spearmanr(sub[f], sub[o])
            corr_rows.append({"Feature": f, "Outcome": o, "N pairs": len(sub),
                              "Spearman rho": rho, "p-value": p})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(ANALYSIS_DIR, "SPEARMAN_CORRELATIONS.csv"), index=False)

    print("\n" + "=" * 90)
    print("SPEARMAN CORRELATION: distance/statistic vs cross-benchmark performance")
    print("(20 off-diagonal pairs; negative rho for distances = larger shift, worse transfer)")
    print("=" * 90)
    print(corr.round(4).to_string(index=False))

    # ---- scatter plots ----------------------------------------
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 4))
    for ax, f in zip(axes, features):
        sub = off[[f, "Balanced Accuracy", "Train Benchmark", "Test Benchmark"]].dropna()
        ax.scatter(sub[f], sub["Balanced Accuracy"])
        for _, r in sub.iterrows():
            ax.annotate(f"{r['Train Benchmark'][-1]}→{r['Test Benchmark'][-1]}",
                        (r[f], r["Balanced Accuracy"]), fontsize=6)
        if len(sub) >= 4:
            rho, p = spearmanr(sub[f], sub["Balanced Accuracy"])
            ax.set_title(f"ρ={rho:.2f}, p={p:.3f}", fontsize=9)
        ax.set_xlabel(f)
        ax.set_ylabel("Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "transfer_vs_distance_scatter.png"), dpi=300)
    plt.close()

    # ---- MMD matrices -----------------------------------------
    for col in ["Embedding MMD", "Pixel MMD"]:
        mat = pairs.pivot(index="Train Benchmark", columns="Test Benchmark",
                          values=col).reindex(index=BENCHMARKS, columns=BENCHMARKS)
        mat.to_csv(os.path.join(ANALYSIS_DIR, f"{col.replace(' ', '_')}_5x5.csv"))
        save_heatmap(mat, None, f"{col} (source train vs target test)", col,
                     os.path.join(ANALYSIS_DIR, f"{col.replace(' ', '_')}_heatmap.png"))


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("ICCAD-12 CROSS-BENCHMARK GENERALISATION (REVISED)")
    print(f"Image size: {IMAGE_SIZE} | Variants: {list(VARIANTS)}")
    print(f"Seeds (full): {SEEDS_FULL} | Seeds (ablation): {SEEDS_ABLATION}")
    print("Benchmarks are NEVER merged: 5 independent models per variant/seed.")
    print("=" * 100)

    all_data = None
    if RUN_TRAINING or RUN_TRANSFER_ANALYSIS:
        all_data = load_all_data()

    if RUN_TRAINING:
        run_all_training(all_data)

    results_df = None
    if RUN_AGGREGATION or RUN_TRANSFER_ANALYSIS:
        results_df = load_all_runs()

    if RUN_AGGREGATION:
        aggregate_main_model(results_df)
        aggregate_ablation_and_gap(results_df)

    if RUN_TRANSFER_ANALYSIS:
        transfer_analysis(all_data, results_df)

    print("\nDONE. Results in:", RESULT_DIR)
