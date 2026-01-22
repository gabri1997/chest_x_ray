# Chest X-ray Multilabel Classification with CNNs and Vision Transformers

Questo progetto affronta il problema della **classificazione multilabel di immagini Chest X-ray** utilizzando sia **reti convoluzionali (DenseNet-121)** sia **Vision Transformer (ViT)**, con l’obiettivo di confrontarne le prestazioni e analizzarne i comportamenti in ambito medicale.

---

## 📌 Obiettivo

Dato un esame radiografico del torace, il modello deve predire **più patologie simultaneamente** (multilabel classification), un setting tipico in ambito clinico.

Il progetto si concentra su:
- Confronto CNN vs Transformer
- Analisi delle metriche multilabel
- Studio di problemi di class imbalance
- Training end-to-end con fine-tuning

---

## 🧠 Modelli Utilizzati

### 1️⃣ DenseNet-121
- Pre-addestrata su ImageNet
- Feature extractor convoluzionale
- Classificatore finale `Linear → BCEWithLogitsLoss`
- Fine-tuning completo di tutti i layer

### 2️⃣ Simple Vision Transformer (ViT)
Implementazione custom di un Vision Transformer semplificato:
- Patch embedding (16×16)
- Self-attention multi-head
- Residual connections + LayerNorm
- CLS token per classificazione globale
- Allenato from scratch sul dataset medicale

---

## 🏗 Architettura ViT (schema concettuale)

```
Immagine
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

## 🧪 Dataset

- Dataset di Chest X-ray con annotazioni multilabel
- Split: Training / Validation / Test
- Dataset fortemente sbilanciato (class imbalance)

---

## 🔄 Data Augmentation & Preprocessing

### Training
- Resize (256)
- Random Resized Crop (224)
- Horizontal Flip
- Normalizzazione ImageNet

### Validation/Test
- Resize (256)
- Center Crop (224)
- Normalizzazione ImageNet

---

## ⚙️ Training Setup

- Loss: `BCEWithLogitsLoss`
- Optimizer:
  - DenseNet → SGD + momentum
  - ViT → AdamW
- Scheduler: StepLR
- Early stopping basato su AUROC macro
- Metriche multilabel con torchmetrics
- Logging con Weights & Biases (wandb)

---

## 📊 Metriche Utilizzate

- Accuracy (micro)
- Precision (micro)
- Recall (micro)
- F1-score (micro)
- F1-score (macro)
- AUROC (macro)

⚠️ L’accuracy non è sufficiente in contesti multilabel sbilanciati.

---

## 📈 Risultati Migliori

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

## 🧠 Analisi dei Risultati

- Accuracy simile → non discriminante
- ViT ottiene AUROC macro più alto
- ViT più sensibile alle classi rare
- DenseNet più conservativa (precision ↑, recall ↓)
- In ambito medicale, alta sensibilità è spesso preferibile

---

## 🚧 Limiti Attuali

- Forte class imbalance
- F1 macro ancora basso
- ViT allenato from scratch

---

## 🚀 Sviluppi Futuri

- Class-weighted BCE / Focal Loss
- Threshold tuning per classi
- Visualizzazione mappe di attenzione
- Pretraining self-supervised
- Ablation study

---

## 🛠 Tecnologie Utilizzate

- PyTorch
- Torchvision
- Torchmetrics
- Vision Transformers
- Weights & Biases
- NVIDIA CUDA

---

## 👨‍💻 Autore

Progetto sviluppato a scopo di studio e ricerca su CNN e Transformer in medical imaging.

---

## 📜 Licenza

Uso accademico / sperimentale.
