# ============================================================
# ICCAD-12 CROSS-BENCHMARK GENERALIZATION
#
# IMPORTANT:
# - The 5 benchmarks are NEVER merged.
# - 5 completely independent models are trained.
# - Each model is trained on ONE benchmark only.
# - Each model is tested on ALL 5 test benchmarks.
#
# REQUIRED OUTPUT:
# - 5 x 5 generalization matrices
# - Accuracy
# - Precision
# - Recall
# - F1
# - Sensitivity
# - Specificity
# - Balanced Accuracy
# - MCC
# - ROC-AUC
# - PR-AUC
# - Confusion matrices
# - Classification reports
# ============================================================


import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

# CHANGE THIS PATH
DATASET_PATH = r"C:\Users\Shrihita\Documents\aiml proj\iccad_official\iccad-official"

BENCHMARKS = [
    "iccad1",
    "iccad2",
    "iccad3",
    "iccad4",
    "iccad5"
]

IMAGE_SIZE = (64, 64)

BATCH_SIZE = 32

EPOCHS = 60

VALIDATION_SPLIT = 0.20

SEED = 42


# Output folders
MODEL_DIR = os.path.join(
    DATASET_PATH,
    "models"
)

RESULT_DIR = os.path.join(
    DATASET_PATH,
    "results"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# 3. LOAD IMAGES FROM ONE BENCHMARK
# ============================================================

def load_images(directory):

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    images = []
    labels = []

    class_mapping = {
        "hotspot": 1,
        "hotspots": 1,
        "hs": 1,
        "train_hs": 1,
        "test_hs": 1,

        "non_hotspot": 0,
        "non-hotspot": 0,
        "nonhotspot": 0,
        "nhs": 0,
        "train_nhs": 0,
        "test_nhs": 0
    }

    for class_name in os.listdir(directory):

        class_path = os.path.join(
            directory,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        class_key = class_name.lower().strip()

        if class_key not in class_mapping:
            print(
                f"Skipping unknown folder: {class_name}"
            )
            continue

        label = class_mapping[class_key]

        print(
            f"Loading {class_name}: class {label}"
        )

        for filename in os.listdir(class_path):

            filepath = os.path.join(
                class_path,
                filename
            )

            try:

                image = tf.keras.utils.load_img(
                    filepath,
                    target_size=IMAGE_SIZE,
                    color_mode="grayscale"
                )

                image = tf.keras.utils.img_to_array(
                    image
                )

                image = image / 255.0

                images.append(image)
                labels.append(label)

            except Exception as e:

                print(
                    f"Could not load {filepath}: {e}"
                )

    if len(images) == 0:
        raise ValueError(
            f"No images found in {directory}"
        )

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


# ============================================================
# 4. LOAD ONE BENCHMARK
# ============================================================

def load_benchmark(benchmark):

    print("\n")
    print("=" * 70)
    print("LOADING:", benchmark)
    print("=" * 70)


    train_path = os.path.join(

        DATASET_PATH,

        benchmark,

        "train"
    )


    test_path = os.path.join(

        DATASET_PATH,

        benchmark,

        "test"
    )


    X_train, y_train = load_images(
        train_path
    )


    X_test, y_test = load_images(
        test_path
    )


    print(
        "Training images:",
        len(X_train)
    )


    print(
        "Training hotspots:",
        np.sum(y_train == 1)
    )


    print(
        "Training non-hotspots:",
        np.sum(y_train == 0)
    )


    print(
        "Testing images:",
        len(X_test)
    )


    print(
        "Testing hotspots:",
        np.sum(y_test == 1)
    )


    print(
        "Testing non-hotspots:",
        np.sum(y_test == 0)
    )


    return {

        "X_train": X_train,
        "y_train": y_train,

        "X_test": X_test,
        "y_test": y_test
    }


# ============================================================
# 5. LOAD ALL FIVE DATASETS
#
# IMPORTANT:
# This does NOT merge them.
#
# They remain in separate dictionary entries.
# ============================================================

def load_all_data():

    all_data = {}


    for benchmark in BENCHMARKS:

        all_data[benchmark] = load_benchmark(
            benchmark
        )


    return all_data


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            mode="horizontal_and_vertical"
        ),

        layers.RandomRotation(
            factor=0.05
        ),

        layers.RandomZoom(
            height_factor=0.05,
            width_factor=0.05
        )

    ],

    name="data_augmentation"
)


# ============================================================
# 7. RESIDUAL BLOCK
# ============================================================

def residual_block(
    x,
    filters,
    dropout_rate
):

    shortcut = x


    # First convolution
    x = layers.Conv2D(

        filters,

        kernel_size=(3, 3),

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)


    # Second convolution
    x = layers.Conv2D(

        filters,

        kernel_size=(3, 3),

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)


    # Match channels for skip connection
    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(

            filters,

            kernel_size=(1, 1),

            padding="same",

            use_bias=False

        )(shortcut)


        shortcut = layers.BatchNormalization()(
            shortcut
        )


    # Residual connection
    x = layers.Add()(
        [x, shortcut]
    )


    x = layers.ReLU()(x)


    x = layers.Dropout(
        dropout_rate
    )(x)


    return x


# ============================================================
# 8. SQUEEZE-AND-EXCITATION ATTENTION
# ============================================================

def se_attention(
    x,
    reduction=8
):

    channels = x.shape[-1]


    # Global information
    attention = layers.GlobalAveragePooling2D()(
        x
    )


    # Compress channels
    attention = layers.Dense(

        max(
            channels // reduction,
            4
        ),

        activation="relu"

    )(attention)


    # Reweight channels
    attention = layers.Dense(

        channels,

        activation="sigmoid"

    )(attention)


    attention = layers.Reshape(

        (1, 1, channels)

    )(attention)


    # Apply attention
    x = layers.Multiply()(
        [x, attention]
    )


    return x


# ============================================================
# 9. IMPROVED CNN ARCHITECTURE
# ============================================================

def build_model():

    inputs = layers.Input(

        shape=(

            IMAGE_SIZE[0],

            IMAGE_SIZE[1],

            1

        )

    )


    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    x = data_augmentation(
        inputs
    )


    # --------------------------------------------------------
    # Initial convolution
    # --------------------------------------------------------

    x = layers.Conv2D(

        32,

        (3, 3),

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)


    # --------------------------------------------------------
    # RESIDUAL BLOCK 1
    # --------------------------------------------------------

    x = residual_block(

        x,

        filters=32,

        dropout_rate=0.05

    )


    x = layers.MaxPooling2D(

        pool_size=(2, 2)

    )(x)


    # --------------------------------------------------------
    # RESIDUAL BLOCK 2
    # --------------------------------------------------------

    x = residual_block(

        x,

        filters=64,

        dropout_rate=0.10

    )


    x = layers.MaxPooling2D(

        pool_size=(2, 2)

    )(x)


    # --------------------------------------------------------
    # RESIDUAL BLOCK 3
    # --------------------------------------------------------

    x = residual_block(

        x,

        filters=128,

        dropout_rate=0.15

    )


    # --------------------------------------------------------
    # SE ATTENTION
    # --------------------------------------------------------

    x = se_attention(x)


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    x = layers.Dense(

        64,

        activation="relu",

        kernel_regularizer=
        regularizers.l2(1e-4)

    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Dropout(
        0.35
    )(x)


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = layers.Dense(

        1,

        activation="sigmoid"

    )(x)


    model = Model(

        inputs,

        outputs,

        name="Residual_Attention_Hotspot_CNN"

    )


    return model


# ============================================================
# 10. FOCAL LOSS
# ============================================================

def focal_loss(
    gamma=2.0,
    alpha=0.75
):

    def loss(
        y_true,
        y_pred
    ):

        y_true = tf.cast(

            y_true,

            tf.float32

        )


        epsilon = tf.keras.backend.epsilon()


        y_pred = tf.clip_by_value(

            y_pred,

            epsilon,

            1.0 - epsilon

        )


        positive_loss = (

            -alpha

            * y_true

            * tf.pow(

                1.0 - y_pred,

                gamma

            )

            * tf.math.log(y_pred)

        )


        negative_loss = (

            -(1.0 - alpha)

            * (1.0 - y_true)

            * tf.pow(

                y_pred,

                gamma

            )

            * tf.math.log(

                1.0 - y_pred

            )

        )


        return tf.reduce_mean(

            positive_loss +

            negative_loss

        )


    return loss


# ============================================================
# 11. CLASS WEIGHTS
# ============================================================

def get_class_weights(y):

    negative_count = np.sum(
        y == 0
    )

    positive_count = np.sum(
        y == 1
    )


    total = len(y)


    weight_0 = (

        total /

        (2.0 * negative_count)

    )


    weight_1 = (

        total /

        (2.0 * positive_count)

    )


    return {

        0: weight_0,

        1: weight_1

    }


# ============================================================
# 12. FIND BEST THRESHOLD
#
# IMPORTANT:
# Threshold is determined using SOURCE VALIDATION DATA ONLY.
#
# Target benchmark test data is NEVER used to choose
# the threshold.
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(

        0.10,

        0.91,

        0.01

    )


    best_threshold = 0.50

    best_score = -1


    for threshold in thresholds:

        predictions = (

            probabilities >= threshold

        ).astype(int)


        score = balanced_accuracy_score(

            y_true,

            predictions

        )


        if score > best_score:

            best_score = score

            best_threshold = threshold


    return (

        best_threshold,

        best_score

    )


# ============================================================
# 13. CALCULATE METRICS
# ============================================================

def calculate_metrics(

    y_true,

    probabilities,

    threshold

):

    predictions = (

        probabilities >= threshold

    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(

        y_true,

        predictions,

        labels=[0, 1]

    ).ravel()


    # Accuracy
    accuracy = accuracy_score(

        y_true,

        predictions

    )


    # Precision
    precision = precision_score(

        y_true,

        predictions,

        zero_division=0

    )


    # Recall
    recall = recall_score(

        y_true,

        predictions,

        zero_division=0

    )


    # F1
    f1 = f1_score(

        y_true,

        predictions,

        zero_division=0

    )


    # Sensitivity
    sensitivity = (

        tp / (tp + fn)

        if (tp + fn) > 0

        else 0

    )


    # Specificity
    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0

    )


    # Balanced Accuracy
    balanced_accuracy = (

        sensitivity +

        specificity

    ) / 2.0


    # MCC
    mcc = matthews_corrcoef(

        y_true,

        predictions

    )


    # ROC-AUC
    try:

        roc_auc = roc_auc_score(

            y_true,

            probabilities

        )

    except:

        roc_auc = np.nan


    # PR-AUC
    try:

        pr_auc = average_precision_score(

            y_true,

            probabilities

        )

    except:

        pr_auc = np.nan


    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Sensitivity": sensitivity,

        "Specificity": specificity,

        "Balanced Accuracy":
            balanced_accuracy,

        "MCC": mcc,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc,

        "TP": tp,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "Threshold": threshold

    }


# ============================================================
# 14. TRAIN ONE INDEPENDENT MODEL
# ============================================================

def train_one_benchmark(

    benchmark,

    benchmark_data

):

    print("\n\n")

    print("#" * 90)

    print(

        "INDEPENDENT EXPERIMENT"

    )

    print(

        f"TRAINING BENCHMARK: {benchmark}"

    )

    print("#" * 90)


    X = benchmark_data["X_train"]

    y = benchmark_data["y_train"]


    # --------------------------------------------------------
    # Split ONLY THIS benchmark's training data
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = (

        train_test_split(

            X,

            y,

            test_size=VALIDATION_SPLIT,

            stratify=y,

            random_state=SEED

        )

    )


    print(

        "\nTraining samples:",

        len(X_train)

    )


    print(

        "Validation samples:",

        len(X_val)

    )


    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = get_class_weights(
        y_train
    )


    print(

        "Class weights:",

        class_weights

    )


    # --------------------------------------------------------
    # Build a COMPLETELY NEW model
    # --------------------------------------------------------

    model = build_model()


    print(

        "\nModel parameters:",

        model.count_params()

    )


    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.AdamW(

            learning_rate=0.001,

            weight_decay=0.0001

        ),

        loss=focal_loss(

            gamma=2.0,

            alpha=0.75

        ),

        metrics=[

            tf.keras.metrics.BinaryAccuracy(

                name="accuracy"

            ),

            tf.keras.metrics.Precision(

                name="precision"

            ),

            tf.keras.metrics.Recall(

                name="recall"

            ),

            tf.keras.metrics.AUC(

                name="auc"

            )

        ]

    )


    # --------------------------------------------------------
    # Save this benchmark's model separately
    # --------------------------------------------------------

    model_path = os.path.join(

        MODEL_DIR,

        f"{benchmark}_independent_model.keras"

    )


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        EarlyStopping(

            monitor="val_auc",

            patience=10,

            mode="max",

            restore_best_weights=True

        ),

        ReduceLROnPlateau(

            monitor="val_auc",

            factor=0.5,

            patience=4,

            mode="max",

            min_lr=1e-6

        ),

        ModelCheckpoint(

            model_path,

            monitor="val_auc",

            mode="max",

            save_best_only=True

        )

    ]


    # --------------------------------------------------------
    # TRAIN
    #
    # ONLY THIS BENCHMARK'S TRAINING DATA IS USED.
    # --------------------------------------------------------

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1

    )


    # --------------------------------------------------------
    # Find threshold using SOURCE VALIDATION ONLY
    # --------------------------------------------------------

    validation_probabilities = model.predict(

        X_val,

        batch_size=BATCH_SIZE,

        verbose=0

    ).ravel()


    best_threshold, validation_ba = (

        find_best_threshold(

            y_val,

            validation_probabilities

        )

    )


    print("\n")

    print(

        "Best validation threshold:",

        round(best_threshold, 3)

    )


    print(

        "Validation Balanced Accuracy:",

        round(validation_ba, 4)

    )


    # --------------------------------------------------------
    # Save training curve
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))


    plt.plot(

        history.history["loss"],

        label="Training Loss"

    )


    plt.plot(

        history.history["val_loss"],

        label="Validation Loss"

    )


    plt.xlabel("Epoch")

    plt.ylabel("Loss")


    plt.title(

        f"{benchmark} - Training Loss"

    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            f"{benchmark}_training_loss.png"

        ),

        dpi=300

    )


    plt.close()


    return (

        model,

        best_threshold

    )


# ============================================================
# 15. TEST ONE MODEL ON ONE BENCHMARK
# ============================================================

def test_model(

    model,

    source_benchmark,

    target_benchmark,

    threshold,

    target_data

):

    X_test = target_data["X_test"]

    y_test = target_data["y_test"]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict(

        X_test,

        batch_size=BATCH_SIZE,

        verbose=0

    ).ravel()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(

        y_test,

        probabilities,

        threshold

    )


    metrics["Train Benchmark"] = (

        source_benchmark

    )


    metrics["Test Benchmark"] = (

        target_benchmark

    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (

        probabilities >= threshold

    ).astype(int)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(

        y_test,

        predictions,

        labels=[0, 1]

    )


    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    plt.figure(figsize=(6, 5))


    plt.imshow(cm)


    plt.title(

        f"Train {source_benchmark} → "
        f"Test {target_benchmark}"

    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.xticks(

        [0, 1],

        [

            "Non-Hotspot",

            "Hotspot"

        ]

    )


    plt.yticks(

        [0, 1],

        [

            "Non-Hotspot",

            "Hotspot"

        ]

    )


    for i in range(2):

        for j in range(2):

            plt.text(

                j,

                i,

                cm[i, j],

                ha="center",

                va="center"

            )


    plt.colorbar()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            f"{source_benchmark}_TO_"
            f"{target_benchmark}_confusion_matrix.png"

        ),

        dpi=300

    )


    plt.close()


    return metrics


# ============================================================
# 16. MAIN EXPERIMENT
# ============================================================

print("\n")

print("=" * 100)

print(
    "ICCAD-12 CROSS-BENCHMARK GENERALIZATION EXPERIMENT"
)

print("=" * 100)

print("\n")

print(
    "IMPORTANT: Benchmarks are NOT being merged."
)

print(
    "Five independent models will be trained."
)

print(
    "Each model uses ONE benchmark for training."
)

print(
    "Each model is tested on ALL FIVE benchmarks."
)

print("\n")


# ============================================================
# LOAD DATA
# ============================================================

all_data = load_all_data()


# ============================================================
# STORE ALL RESULTS
# ============================================================

all_results = []


# ============================================================
# TRAIN 5 INDEPENDENT MODELS
# ============================================================

for source_benchmark in BENCHMARKS:


    # --------------------------------------------------------
    # NEW model trained ONLY on this benchmark
    # --------------------------------------------------------

    model, threshold = train_one_benchmark(

        source_benchmark,

        all_data[source_benchmark]

    )


    # --------------------------------------------------------
    # TEST THIS MODEL ON ALL FIVE BENCHMARKS
    # --------------------------------------------------------

    print("\n")

    print("-" * 90)

    print(

        f"TESTING MODEL TRAINED ON {source_benchmark}"

    )

    print("-" * 90)


    for target_benchmark in BENCHMARKS:


        print(

            f"\n{source_benchmark} → "
            f"{target_benchmark}"

        )


        metrics = test_model(

            model,

            source_benchmark,

            target_benchmark,

            threshold,

            all_data[target_benchmark]

        )


        all_results.append(metrics)


        print(

            f"Accuracy: "
            f"{metrics['Accuracy']:.4f}"

        )

        print(

            f"Precision: "
            f"{metrics['Precision']:.4f}"

        )

        print(

            f"Recall: "
            f"{metrics['Recall']:.4f}"

        )

        print(

            f"F1: "
            f"{metrics['F1']:.4f}"

        )

        print(

            f"Sensitivity: "
            f"{metrics['Sensitivity']:.4f}"

        )

        print(

            f"Specificity: "
            f"{metrics['Specificity']:.4f}"

        )

        print(

            f"Balanced Accuracy: "
            f"{metrics['Balanced Accuracy']:.4f}"

        )

        print(

            f"MCC: "
            f"{metrics['MCC']:.4f}"

        )

        print(

            f"ROC-AUC: "
            f"{metrics['ROC-AUC']:.4f}"

        )

        print(

            f"PR-AUC: "
            f"{metrics['PR-AUC']:.4f}"

        )


# ============================================================
# 17. COMPLETE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(

    all_results

)


results_df = results_df[

    [

        "Train Benchmark",

        "Test Benchmark",

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "Sensitivity",

        "Specificity",

        "Balanced Accuracy",

        "MCC",

        "ROC-AUC",

        "PR-AUC",

        "TP",

        "TN",

        "FP",

        "FN",

        "Threshold"

    ]

]


# ============================================================
# SAVE ALL 25 EXPERIMENTS
# ============================================================

results_df.to_csv(

    os.path.join(

        RESULT_DIR,

        "ALL_25_CROSS_BENCHMARK_RESULTS.csv"

    ),

    index=False

)


# ============================================================
# 18. PRINT COMPLETE RESULTS
# ============================================================

print("\n\n")

print("=" * 120)

print(
    "ALL 25 TRAIN → TEST RESULTS"
)

print("=" * 120)


print(

    results_df.round(4).to_string(

        index=False

    )

)


# ============================================================
# 19. CREATE 5 x 5 MATRICES
# ============================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "Sensitivity",

    "Specificity",

    "Balanced Accuracy",

    "MCC",

    "ROC-AUC",

    "PR-AUC"

]


for metric in metrics:


    matrix = results_df.pivot(

        index="Train Benchmark",

        columns="Test Benchmark",

        values=metric

    )


    # --------------------------------------------------------
    # Save matrix
    # --------------------------------------------------------

    safe_name = metric.replace(

        " ",

        "_"

    )


    matrix.to_csv(

        os.path.join(

            RESULT_DIR,

            f"{safe_name}_5x5_matrix.csv"

        )

    )


    # --------------------------------------------------------
    # Print matrix
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 100)

    print(

        f"{metric.upper()} - 5 × 5 GENERALIZATION MATRIX"

    )

    print("=" * 100)


    print(

        matrix.round(4)

    )


# ============================================================
# 20. BALANCED ACCURACY HEATMAP
# ============================================================

ba_matrix = results_df.pivot(

    index="Train Benchmark",

    columns="Test Benchmark",

    values="Balanced Accuracy"

)


plt.figure(figsize=(8, 7))


plt.imshow(

    ba_matrix.values,

    aspect="auto"

)


plt.colorbar(

    label="Balanced Accuracy"

)


plt.xticks(

    range(5),

    BENCHMARKS

)


plt.yticks(

    range(5),

    BENCHMARKS

)


plt.xlabel(
    "Test Benchmark"
)


plt.ylabel(
    "Training Benchmark"
)


plt.title(

    "ICCAD-12 Cross-Benchmark "
    "Balanced Accuracy"

)


# Add numbers
for i in range(5):

    for j in range(5):

        value = ba_matrix.iloc[i, j]


        plt.text(

            j,

            i,

            f"{value:.3f}",

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    os.path.join(

        RESULT_DIR,

        "Balanced_Accuracy_5x5_Heatmap.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 21. F1 HEATMAP
# ============================================================

f1_matrix = results_df.pivot(

    index="Train Benchmark",

    columns="Test Benchmark",

    values="F1"

)


plt.figure(figsize=(8, 7))


plt.imshow(

    f1_matrix.values,

    aspect="auto"

)


plt.colorbar(

    label="F1 Score"

)


plt.xticks(

    range(5),

    BENCHMARKS

)


plt.yticks(

    range(5),

    BENCHMARKS

)


plt.xlabel(
    "Test Benchmark"
)


plt.ylabel(
    "Training Benchmark"
)


plt.title(

    "ICCAD-12 Cross-Benchmark F1 Score"

)


for i in range(5):

    for j in range(5):

        value = f1_matrix.iloc[i, j]


        plt.text(

            j,

            i,

            f"{value:.3f}",

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    os.path.join(

        RESULT_DIR,

        "F1_5x5_Heatmap.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 22. DIAGONAL RESULTS
#
# B1 → B1
# B2 → B2
# B3 → B3
# B4 → B4
# B5 → B5
#
# These are within-benchmark results.
# ============================================================

diagonal_rows = []


for benchmark in BENCHMARKS:

    row = results_df[

        (

            results_df[
                "Train Benchmark"
            ]

            == benchmark

        )

        &

        (

            results_df[
                "Test Benchmark"
            ]

            == benchmark

        )

    ]


    diagonal_rows.append(

        row.iloc[0]

    )


diagonal_df = pd.DataFrame(

    diagonal_rows

)


print("\n\n")

print("=" * 100)

print(

    "DIAGONAL / WITHIN-BENCHMARK RESULTS"

)

print("=" * 100)


print(

    diagonal_df[

        [

            "Train Benchmark",

            "Test Benchmark",

            "Accuracy",

            "Precision",

            "Recall",

            "F1",

            "Sensitivity",

            "Specificity",

            "Balanced Accuracy",

            "MCC",

            "ROC-AUC",

            "PR-AUC"

        ]

    ].round(4).to_string(

        index=False

    )

)


# ============================================================
# 23. OFF-DIAGONAL RESULTS
#
# These are the actual CROSS-BENCHMARK GENERALIZATION
# experiments.
# ============================================================

off_diagonal_df = results_df[

    results_df[
        "Train Benchmark"
    ]

    !=

    results_df[
        "Test Benchmark"
    ]

]


print("\n\n")

print("=" * 100)

print(

    "OFF-DIAGONAL / CROSS-BENCHMARK RESULTS"

)

print("=" * 100)


print(

    off_diagonal_df[

        [

            "Train Benchmark",

            "Test Benchmark",

            "Accuracy",

            "Precision",

            "Recall",

            "F1",

            "Sensitivity",

            "Specificity",

            "Balanced Accuracy",

            "MCC",

            "ROC-AUC",

            "PR-AUC"

        ]

    ].round(4).to_string(

        index=False

    )

)


# ============================================================
# 24. AVERAGE WITHIN-BENCHMARK PERFORMANCE
# ============================================================

print("\n\n")

print("=" * 100)

print(

    "AVERAGE WITHIN-BENCHMARK PERFORMANCE"

)

print("=" * 100)


for metric in metrics:

    value = diagonal_df[

        metric

    ].mean()


    print(

        f"{metric:25s}: "
        f"{value:.4f}"

    )


# ============================================================
# 25. AVERAGE CROSS-BENCHMARK GENERALIZATION
# ============================================================

print("\n\n")

print("=" * 100)

print(

    "AVERAGE CROSS-BENCHMARK GENERALIZATION"

)

print("=" * 100)


for metric in metrics:

    value = off_diagonal_df[

        metric

    ].mean()


    print(

        f"{metric:25s}: "
        f"{value:.4f}"

    )


# ============================================================
# 26. SOURCE-BENCHMARK ANALYSIS
#
# Average performance of each model when transferred to
# the OTHER FOUR benchmarks.
# ============================================================

source_summary = (

    off_diagonal_df

    .groupby(

        "Train Benchmark"

    )[

        metrics

    ]

    .mean()

)


print("\n\n")

print("=" * 100)

print(

    "AVERAGE TRANSFER PERFORMANCE BY TRAINING BENCHMARK"

)

print("=" * 100)


print(

    source_summary.round(4)

)


source_summary.to_csv(

    os.path.join(

        RESULT_DIR,

        "SOURCE_TRANSFER_SUMMARY.csv"

    )

)


# ============================================================
# 27. TARGET-BENCHMARK ANALYSIS
#
# Average performance when each benchmark is used as the
# unseen target.
# ============================================================

target_summary = (

    off_diagonal_df

    .groupby(

        "Test Benchmark"

    )[

        metrics

    ]

    .mean()

)


print("\n\n")

print("=" * 100)

print(

    "AVERAGE GENERALIZATION PERFORMANCE BY TARGET BENCHMARK"

)

print("=" * 100)


print(

    target_summary.round(4)

)


target_summary.to_csv(

    os.path.join(

        RESULT_DIR,

        "TARGET_GENERALIZATION_SUMMARY.csv"

    )

)


# ============================================================
# 28. TRANSFER PAIRS
#
# Shows all off-diagonal pairs ordered by Balanced Accuracy.
#
# This is for identifying strong/weak transferability.
# ============================================================

pair_analysis = off_diagonal_df.sort_values(

    by="Balanced Accuracy",

    ascending=False

)


print("\n\n")

print("=" * 100)

print(

    "TRAIN → TEST PAIRS ORDERED BY BALANCED ACCURACY"

)

print("=" * 100)


print(

    pair_analysis[

        [

            "Train Benchmark",

            "Test Benchmark",

            "Balanced Accuracy",

            "F1",

            "Precision",

            "Recall"

        ]

    ].round(4).to_string(

        index=False

    )

)


pair_analysis.to_csv(

    os.path.join(

        RESULT_DIR,

        "TRANSFER_PAIR_ANALYSIS.csv"

    ),

    index=False

)


# ============================================================
# 29. FINISHED
# ============================================================

print("\n\n")

print("=" * 100)

print(

    "EXPERIMENT COMPLETE"

)

print("=" * 100)

print("\n")

print(

    "5 independent models were trained."

)

print(

    "25 independent train → test evaluations were performed."

)

print(

    "The five benchmarks were NEVER merged."

)

print("\n")

print(

    "Results saved to:"

)

print(

    RESULT_DIR

)

print("\n")

print(

    "Main file:"

)

print(

    os.path.join(

        RESULT_DIR,

        "ALL_25_CROSS_BENCHMARK_RESULTS.csv"

    )

)
