# Chest X-ray Multilabel Classification with CNNs and Vision Transformers

This project addresses the problem of **multilabel classification of Chest X-ray images** using both **convolutional neural networks (DenseNet-121)** and **Vision Transformers (ViT)**, with the goal of comparing their performance and analyzing their behavior in the medical imaging domain.

---

## Objective

Given a chest X-ray image, the model must predict **multiple pathologies simultaneously** (multilabel classification), a setting that is typical in clinical practice.

The project focuses on:
- CNN vs Transformer comparison
- Analysis of multilabel evaluation metrics
- Study of class imbalance issues
- End-to-end training with fine-tuning

---

## Models Used

### DenseNet-121
- Pretrained on ImageNet
- Convolutional feature extractor
- Final classifier: `Linear → BCEWithLogitsLoss`
- Full fine-tuning of all layers

### Simple Vision Transformer (ViT)
Custom implementation of a simplified Vision Transformer:
- Patch embedding (16×16)
- Multi-head self-attention
- Residual connections + LayerNorm
- CLS token for global classification
- Trained from scratch on the medical dataset

---

## ViT Architecture (Conceptual Diagram)

```
Image
 ↓
Patch Embedding
 ↓
[ Transformer Block × N ]
   ├── LayerNorm
   ├── Multi-Head Self-Attention
   ├── Skip Connection
   ├── LayerNorm
   ├── FeedForward (MLP)
   └── Skip Connection
 ↓
CLS Token
 ↓
Linear Head
 ↓
Sigmoid (multilabel)
```

---

## Dataset

- Chest X-ray dataset with multilabel annotations
- Splits: Training / Validation / Test
- Highly imbalanced dataset (class imbalance)

---

## Data Augmentation & Preprocessing

### Training
- Resize (256)
- Random Resized Crop (224)
- Horizontal Flip
- ImageNet normalization

### Validation/Test
- Resize (256)
- Center Crop (224)
- ImageNet normalization

---

## Training Setup

- Loss: `BCEWithLogitsLoss`
- Optimizer:
  - DenseNet → SGD + momentum
  - ViT → AdamW
- Scheduler: StepLR
- Early stopping based on macro AUROC
- Multilabel metrics via torchmetrics
- Logging with Weights & Biases (wandb)

---

## Evaluation Metrics

- Accuracy (micro)
- Precision (micro)
- Recall (micro)
- F1-score (micro)
- F1-score (macro)
- AUROC (macro)

Accuracy alone is not sufficient in imbalanced multilabel settings.

---

## Best Results

### DenseNet-121
```
Accuracy (micro):   0.9303
Precision (micro):  0.6975
Recall (micro):     0.3073
F1 (micro):         0.4266
F1 (macro):         0.0487
AUROC (macro):      0.7021
Val loss:           0.1993
```

### Vision Transformer (SimpleViT)
```
Accuracy (micro):   0.9294
Precision (micro):  0.6728
Recall (micro):     0.3165
F1 (micro):         0.4305
F1 (macro):         0.0575
AUROC (macro):      0.7309
Val loss:           0.1981
```

---

## Results Analysis

- Similar accuracy → not discriminative
- ViT achieves higher macro AUROC
- ViT is more sensitive to rare classes
- DenseNet is more conservative (higher precision, lower recall)
- In medical applications, higher sensitivity is often preferable

---

## Current Limitations

- Strong class imbalance
- Macro F1-score still low
- ViT trained from scratch

---

## Future Work

- Class-weighted BCE / Focal Loss
- Per-class threshold tuning
- Attention map visualization
- Self-supervised pretraining
- Ablation studies

---

## Technologies Used

- PyTorch
- Torchvision
- Torchmetrics
- Vision Transformers
- Weights & Biases
- NVIDIA CUDA
