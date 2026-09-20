# Deep Learning-Based Lithography Hotspot Detection

## Overview

This project investigates **deep learning-based lithography hotspot detection** using the five **ICCAD-12 benchmark datasets**.

The main objective is to study how well a CNN trained on one benchmark can generalize to the other benchmark distributions. To achieve this, both a **Baseline CNN** and an **Improved/Proposed CNN** are developed and evaluated using a complete **5 × 5 cross-benchmark evaluation framework**.

Each model is trained independently on one ICCAD benchmark and tested on all five benchmarks, resulting in **25 train-test combinations**.

---

## Project Objectives

* Detect lithography **hotspot (HS)** and **non-hotspot (NHS)** patterns.
* Develop a baseline CNN for hotspot classification.
* Improve the baseline using residual learning and SE attention.
* Address the class imbalance present in the ICCAD-12 datasets.
* Evaluate within-benchmark performance.
* Analyse cross-benchmark generalization.
* Compare the baseline and proposed models using multiple evaluation metrics.

---

## Dataset

The experiments use the five ICCAD-12 benchmark datasets:

* ICCAD-1
* ICCAD-2
* ICCAD-3
* ICCAD-4
* ICCAD-5

The classification task is binary:

| Class             | Label |
| ----------------- | ----: |
| Hotspot (HS)       |     1 |
| Non-Hotspot (NHS)  |     0 |

Each benchmark contains training and testing data consisting of hotspot and non-hotspot layout patterns.

Because the datasets are highly imbalanced, evaluation is not based only on conventional accuracy.

---

## Experimental Methodology

Five independent models are trained:

```text
Model 1 → Trained on ICCAD-1 → Tested on ICCAD-1 to ICCAD-5
Model 2 → Trained on ICCAD-2 → Tested on ICCAD-1 to ICCAD-5
Model 3 → Trained on ICCAD-3 → Tested on ICCAD-1 to ICCAD-5
Model 4 → Trained on ICCAD-4 → Tested on ICCAD-1 to ICCAD-5
Model 5 → Trained on ICCAD-5 → Tested on ICCAD-1 to ICCAD-5
```

This produces:

```text
5 training benchmarks × 5 testing benchmarks = 25 experiments
```

The diagonal entries represent **within-benchmark performance**, while the off-diagonal entries represent **cross-benchmark generalization**.

---

# Baseline CNN

The baseline model is a conventional CNN trained from scratch.

### Input

```text
Grayscale image
128 × 128 pixels
1 input channel
```

### Architecture

```text
Input
  ↓
Conv2D (1 → 32) + ReLU
  ↓
MaxPool (2 × 2)
  ↓
Conv2D (32 → 64) + ReLU
  ↓
MaxPool (2 × 2)
  ↓
Conv2D (64 → 128) + ReLU
  ↓
MaxPool (2 × 2)
  ↓
Flatten
  ↓
Fully Connected (128 × 16 × 16 → 128)
  ↓
ReLU
  ↓
Dropout (0.5)
  ↓
Fully Connected (128 → 2)
  ↓
Output
```

The two output classes are:

```text
0 → NHS
1 → HS
```

The baseline CNN serves as the reference model for evaluating the improvements introduced in the proposed model.

---

# Improved / Proposed CNN

The proposed model improves the baseline using:

* Residual learning
* Squeeze-and-Excitation (SE) attention
* Data augmentation
* Class weighting
* Focal loss
* Validation-based decision threshold selection
* Global Average Pooling

### Architecture

```text
Input
64 × 64 × 1
   ↓
Conv2D
32 filters, 3 × 3
   ↓
Residual Block
32 filters
   ↓
SE Attention
   ↓
Residual Block
64 filters
   ↓
SE Attention
   ↓
Residual Block
128 filters
   ↓
SE Attention
   ↓
Global Average Pooling
   ↓
Dense
64 neurons
   ↓
Dropout
p = 0.35
   ↓
Sigmoid
   ↓
Hotspot Probability
```

The proposed architecture uses residual connections to improve feature learning and SE attention to adaptively emphasize informative feature channels.

---

## Handling Class Imbalance

The ICCAD-12 datasets contain substantially more non-hotspot samples than hotspot samples.

The proposed model therefore uses:

### Class Weighting

Class weights are calculated according to the distribution of samples in the training dataset.

### Focal Loss

Focal loss is used with:

```text
γ = 2
α = 0.75
```

This gives greater importance to difficult and minority-class samples.

### Validation-Based Threshold

Instead of always using a classification threshold of `0.5`, the proposed model determines the decision threshold using the source validation dataset.

The selected threshold is then fixed when evaluating all target benchmarks.

---

# Evaluation Metrics

The models are evaluated using multiple metrics:

* Accuracy
* Precision
* Recall
* Specificity
* F1-score
* Balanced Accuracy
* Confusion Matrix

Balanced accuracy is particularly important because of the severe class imbalance between hotspot and non-hotspot samples.

---

# Results

## Mean Balanced Accuracy

| Model        | Mean Balanced Accuracy |
| ------------ | ---------------------: |
| Baseline CNN |             **65.58%** |
| Proposed CNN |             **69.16%** |

The proposed CNN increases the overall mean balanced accuracy from **65.58% to 69.16%**.

---

## Within-Benchmark Performance

The mean diagonal balanced accuracy is:

| Model        | Mean Diagonal Balanced Accuracy |
| ------------ | ------------------------------: |
| Baseline CNN |                      **84.68%** |
| Proposed CNN |                      **87.82%** |

---

## Cross-Benchmark Performance

Considering only the 20 off-diagonal train-test combinations:

| Model        | Mean Cross-Benchmark Balanced Accuracy |
| ------------ | -------------------------------------: |
| Baseline CNN |                             **60.80%** |
| Proposed CNN |                             **64.49%** |

This evaluation highlights the difference between performance on the training benchmark and generalization to other benchmark distributions.

---

## Baseline CNN — 5 × 5 Balanced Accuracy

| Train / Test | ICCAD-1 | ICCAD-2 | ICCAD-3 | ICCAD-4 | ICCAD-5 |
| ------------ | ------: | ------: | ------: | ------: | ------: |
| **ICCAD-1**  |   83.35 |   61.23 |   62.38 |   64.09 |   62.95 |
| **ICCAD-2**  |   50.00 |   82.06 |   50.57 |   56.42 |   50.00 |
| **ICCAD-3**  |   71.42 |   82.10 |   96.09 |   76.47 |   98.49 |
| **ICCAD-4**  |   56.01 |   69.23 |   50.03 |   86.29 |   49.99 |
| **ICCAD-5**  |   50.00 |   50.10 |   54.59 |   50.00 |   75.60 |

---

## Proposed CNN — 5 × 5 Balanced Accuracy

| Train / Test | ICCAD-1 | ICCAD-2 | ICCAD-3 | ICCAD-4 | ICCAD-5 |
| ------------ | ------: | ------: | ------: | ------: | ------: |
| **ICCAD-1**  |   81.33 |   68.29 |   64.55 |   67.37 |   65.19 |
| **ICCAD-2**  |   56.35 |   85.05 |   49.98 |   52.10 |   51.22 |
| **ICCAD-3**  |   60.66 |   88.31 |   94.37 |   82.11 |   92.88 |
| **ICCAD-4**  |   61.00 |   82.15 |   50.09 |   91.31 |   50.00 |
| **ICCAD-5**  |   54.49 |   55.33 |   79.27 |   58.46 |   87.02 |

---

# Key Observations

The experiments show that cross-benchmark performance is highly dependent on the source and target benchmark.

For example:

```text
ICCAD-3 → ICCAD-5 = 92.88%
ICCAD-3 → ICCAD-2 = 88.31%
ICCAD-4 → ICCAD-2 = 82.15%
ICCAD-3 → ICCAD-4 = 82.11%
ICCAD-5 → ICCAD-3 = 79.27%
```

Some transfers remain close to 50% balanced accuracy, indicating limited discrimination for those particular source-target combinations.

The results also demonstrate that transferability is **direction-dependent**. For example:

```text
ICCAD-3 → ICCAD-2 = 88.31%
ICCAD-2 → ICCAD-3 = 49.98%
```

Therefore, training and testing benchmark distributions can have a significant effect on model performance.

The proposed model improves the overall mean balanced accuracy, but the improvement is not uniform across every individual source-target pair.

---

# Project Files

The repository contains the following main files:

```text
├── baseline_code
├── improved_code
├── project_presentation.pptx
└── README.md
```

### `baseline_code`

Contains the implementation of the baseline CNN used as the reference model.

### `improved_code`

Contains the implementation of the proposed CNN with residual learning, SE attention, imbalance-aware training and validation-based threshold selection.

### `project_presentation.pptx`

Contains the project presentation covering the problem statement, methodology, architectures, experiments and results.

---

# Requirements

The models are implemented using Python and PyTorch.

Required libraries include:

```text
Python
PyTorch
Torchvision
NumPy
Pandas
Scikit-learn
Matplotlib
```

A GPU-enabled environment such as **Google Colab** is recommended for faster training.

---

# Running the Project

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

## 2. Prepare the Dataset

Place the ICCAD-12 benchmark datasets according to the directory structure expected by the code.

The five benchmarks should be available as:

```text
iccad1/
iccad2/
iccad3/
iccad4/
iccad5/
```

Each benchmark contains hotspot and non-hotspot training/testing data.

## 3. Run the Baseline Model

Run the baseline CNN code to:

* Train the model
* Evaluate it on ICCAD-1 to ICCAD-5
* Generate confusion matrices
* Calculate classification metrics
* Produce the 5 × 5 evaluation results

## 4. Run the Improved Model

Run the improved CNN code to perform the same cross-benchmark evaluation using the proposed architecture.

The resulting metrics can then be compared with the baseline model.

---

# Experimental Workflow

```text
                 ICCAD-12
                     │
        ┌────────────┼────────────┐
        │            │            │
     ICCAD-1      ICCAD-2      ... ICCAD-5
        │            │                │
        └────────────┼────────────────┘
                     │
              Train Independent
                  CNN Models
                     │
                     ↓
             Cross-Benchmark
                 Testing
                     │
                     ↓
               5 × 5 Matrix
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      Baseline CNN         Proposed CNN
          │                     │
          └──────────┬──────────┘
                     ↓
              Performance
                Comparison
```

---

# Conclusion

This project presents a systematic evaluation of deep learning-based lithography hotspot detection across the five ICCAD-12 benchmarks.

The baseline CNN provides a reference model, while the proposed CNN introduces residual learning, Squeeze-and-Excitation attention, data augmentation, class weighting, focal loss and validation-based threshold selection.

The complete **5 × 5 evaluation framework** allows both within-benchmark performance and cross-benchmark generalization to be analysed.

The proposed model achieves:

```text
Mean Balanced Accuracy
Baseline  : 65.58%
Proposed  : 69.16%

Mean Diagonal Balanced Accuracy
Baseline  : 84.68%
Proposed  : 87.82%

Mean Off-Diagonal Balanced Accuracy
Baseline  : 60.80%
Proposed  : 64.49%
```

The results demonstrate that benchmark distribution has a significant influence on lithography hotspot detection performance and that evaluating only on the training benchmark does not fully characterize model generalization.

---

# Authors
**Madhumitha M**
**Katakam Shrihita**
**Dinesh Kumar**

Department of Electronics Engineering and VLSI Design
Vellore Institute of Technology, Chennai
Chennai, India

---

## Presentation

The complete project presentation is available in:

```text
project_presentation.pptx
```

The presentation contains the project motivation, methodology, baseline and proposed architectures, experimental results and conclusions.
