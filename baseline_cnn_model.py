import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = "/content/iccad_data/iccad-official"

TRAIN_BENCHMARK = "iccad1"

TEST_BENCHMARKS = [
    "iccad1",
    "iccad2",
    "iccad3",
    "iccad4",
    "iccad5"
]

SAVE_PATH = f"/content/baseline_cnn_{TRAIN_BENCHMARK}.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (128, 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 2. Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


# ==========================================
# 3. Load ICCAD-1 Training Dataset
# ==========================================

train_path = os.path.join(
    BASE_DIR,
    TRAIN_BENCHMARK,
    "train"
)

dataset = datasets.ImageFolder(
    train_path,
    transform=transform,
    target_transform=lambda x: 1 - x
)

print("\n--- TRAINING DATA ---")
print("Benchmark:", TRAIN_BENCHMARK)
print("Original classes:", dataset.classes)
print("Class mapping:")
print("HS  = 1")
print("NHS = 0")
print("Total images:", len(dataset))

labels = np.array(dataset.targets)

print("HS samples :", np.sum(labels == 1))
print("NHS samples:", np.sum(labels == 0))


# ==========================================
# 4. Train / Validation Split
# ==========================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# 5. Baseline CNN
# ==========================================

class BaselineCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = BaselineCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. Train CNN on ICCAD-1
# ==========================================

print("\n" + "=" * 55)
print("TRAINING BASELINE CNN ON ICCAD-1")
print("=" * 55)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_acc = 100 * correct / total


    # ---------- Validation ----------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"| Train Acc: {train_acc:.2f}% "
        f"| Val Acc: {val_acc:.2f}%"
    )


# ==========================================
# 7. Save Model
# ==========================================

torch.save(
    model.state_dict(),
    SAVE_PATH
)

print("\nModel saved:", SAVE_PATH)


# ==========================================
# 8. Evaluation Function
# ==========================================

def evaluate_benchmark(model, data_path, bm_name):

    test_dataset = datasets.ImageFolder(
        data_path,
        transform=transform,
        target_transform=lambda x: 1 - x
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, preds = torch.max(
                outputs,
                1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                labels.numpy()
            )

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)


    # ======================================
    # HS = 1
    # NHS = 0
    #
    #              Predicted
    #              HS     NHS
    # Actual HS    TP     FN
    # Actual NHS   FP     TN
    # ======================================

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=[1, 0]
    )

    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]


    # Metrics

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0
    )

    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else 0
    )

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2


    # ======================================
    # Print Results
    # ======================================

    print("\n" + "=" * 55)
    print("TEST BENCHMARK:", bm_name)
    print("=" * 55)

    print("\nConfusion Matrix")
    print("              Predicted")
    print("              HS    NHS")
    print(f"Actual HS     {TP}    {FN}")
    print(f"Actual NHS    {FP}    {TN}")

    print("\nMetrics")

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Precision         : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall/Sensitivity: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Specificity       : "
        f"{specificity * 100:.2f}%"
    )

    print(
        f"F1-score          : "
        f"{f1 * 100:.2f}%"
    )


    return {
        "Train": "ICCAD-1",
        "Test": bm_name,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "Balanced Accuracy": balanced_accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "Specificity": specificity * 100,
        "F1-score": f1 * 100
    }


# ==========================================
# 9. Test ICCAD-1 to ICCAD-5
# ==========================================

results = []

print("\n" + "#" * 60)
print("CROSS-BENCHMARK TESTING")
print("TRAINED ON ICCAD-1")
print("#" * 60)

for bm in TEST_BENCHMARKS:

    test_path = os.path.join(
        BASE_DIR,
        bm,
        "test"
    )

    if not os.path.exists(test_path):

        print(
            f"\nWARNING: Test path not found: {test_path}"
        )

        continue

    result = evaluate_benchmark(
        model,
        test_path,
        bm
    )

    results.append(result)


# ==========================================
# 10. Final Results Table
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n" + "=" * 80)
print("FINAL BASELINE CNN-1 RESULTS")
print("=" * 80)

display(results_df)


# ==========================================
# 11. Save Results
# ==========================================

results_df.to_csv(
    "/content/baseline_iccad1_results.csv",
    index=False
)

print("\nResults saved to:")
print("/content/baseline_iccad1_results.csv")
