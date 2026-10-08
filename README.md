# Lung Histopathology Image Classification Using GhostNet-100 and Lightweight Transformer

## 📌 Project Overview

Lung histopathology images contain complex microscopic tissue patterns that can be difficult to distinguish visually. In particular, **adenocarcinoma and squamous cell carcinoma may exhibit overlapping morphological patterns**, making accurate classification challenging.

This project focuses on developing an efficient deep learning model for classifying lung histopathology images into three categories:

* **Normal**
* **Adenocarcinoma**
* **Squamous Cell Carcinoma**

The project starts with **GhostNet-100 as the baseline model** and improves it by integrating a **lightweight Transformer module** to capture global relationships between different regions of the image.

---

## 🎯 Objective

The main objective is to develop a **lightweight and efficient hybrid CNN-Transformer model** that can learn both:

* **Local visual features** such as cells, nuclei, edges, and tissue textures.
* **Global relationships** between different regions of the histopathology image.

### Proposed Approach

> **GhostNet-100 + Lightweight Transformer**

The CNN extracts efficient local features, while the Transformer models relationships between the extracted features.

---

## 🧠 Base Model – GhostNet-100

**GhostNet-100** is a lightweight convolutional neural network designed to generate useful feature maps with reduced computational cost.

The base model processes the input image through Ghost modules and bottleneck blocks to extract hierarchical visual features.

### What GhostNet-100 learns

The CNN can identify features such as:

* Cell structures
* Nuclei
* Tissue patterns
* Edges
* Textures
* Local morphological features

### Base Model Flow

```text
Input Histopathology Image
            ↓
       GhostNet-100
            ↓
     Ghost Modules
            ↓
     Bottleneck Blocks
            ↓
     Feature Extraction
            ↓
       Feature Maps
            ↓
      Classification
            ↓
      3-Class Output
```

### Limitation of the Base Model

Although GhostNet-100 is efficient at extracting **local features**, CNN-based feature extraction has limitations in explicitly modeling **long-range relationships between distant image regions**.

For histopathology images, information from multiple tissue regions can be important for making a classification decision.

This motivated the integration of a lightweight Transformer.

---

## 🚀 Proposed Improvement

### GhostNet-100 + 1-Layer Lightweight Transformer

The proposed model extends GhostNet-100 by adding a **lightweight Transformer encoder** after CNN feature extraction.

```text
                 Input Image
                      │
                      ▼
                GhostNet-100
                      │
                      ▼
              Local Feature Maps
                      │
                      ▼
               Feature Tokens
                      │
                      ▼
        ┌──────────────────────────┐
        │ Lightweight Transformer  │
        │                          │
        │  1 Transformer Layer     │
        │  4 Attention Heads       │
        │  Embedding Dimension 256 │
        │  Feed Forward = 512      │
        └──────────────────────────┘
                      │
                      ▼
             Global Feature
              Representation
                      │
                      ▼
             Classification Head
                      │
                      ▼
        Normal / Adenocarcinoma /
        Squamous Cell Carcinoma
```

---

## 🏗️ Model Architecture

![GhostNet-100 + Lightweight Transformer Architecture](architecture.png)

The architecture consists of two major components:

### 1. GhostNet-100 – Local Feature Extraction

The input histopathology image is first processed by GhostNet-100.

It extracts local visual patterns such as:

```text
Image
  ↓
Convolutional Feature Extraction
  ↓
Ghost Modules
  ↓
Bottleneck Blocks
  ↓
Feature Maps
```

These feature maps contain important information about the microscopic structures present in the image.

### 2. Lightweight Transformer – Global Feature Modeling

The extracted feature maps are converted into feature tokens and passed to the Transformer.

The Transformer uses **self-attention** to learn relationships between different feature regions.

Instead of considering each region independently, the attention mechanism allows the model to determine which regions are important to each other.

```text
Feature Tokens
      ↓
Multi-Head Self-Attention
      ↓
Relationship Modeling
      ↓
Feed-Forward Network
      ↓
Enhanced Feature Representation
```

---

## 🔍 Why Combine CNN and Transformer?

The two components have complementary strengths.

| Component        | Main Role                                    |
| ---------------- | -------------------------------------------- |
| **GhostNet-100** | Extracts local visual features               |
| **Transformer**  | Learns relationships between feature regions |
| **Hybrid Model** | Combines local and global information        |

### GhostNet-100

Focuses mainly on:

* Cells
* Nuclei
* Edges
* Textures
* Local tissue structures

### Transformer

Focuses on:

* Relationships between different regions
* Global contextual information
* Long-range dependencies
* Important feature interactions

Therefore:

```text
Local Features
      +
Global Relationships
      ↓
Enhanced Feature Representation
      ↓
Classification
```

---

## ⚙️ Transformer Configuration

The proposed experiment uses a lightweight Transformer with the following configuration:

| Parameter              |   Value |
| ---------------------- | ------: |
| Transformer Layers     |   **1** |
| Embedding Dimension    | **256** |
| Attention Heads        |   **4** |
| Feed-Forward Dimension | **512** |
| Dropout                | **0.1** |

### Why only one Transformer layer?

A single Transformer layer keeps the model relatively lightweight while still allowing the architecture to introduce self-attention and global relationship modeling.

The objective is not simply to make the model larger, but to improve feature representation while maintaining computational efficiency.

---

## 🔄 How the Proposed Model Works

The complete process is:

```text
1. Input Image
       ↓
2. GhostNet-100
       ↓
3. Extract Local Feature Maps
       ↓
4. Convert Features into Tokens
       ↓
5. Lightweight Transformer
       ↓
6. Multi-Head Self-Attention
       ↓
7. Global Feature Representation
       ↓
8. Classification Head
       ↓
9. Predicted Lung Histopathology Class
```

### Step-by-step

**Step 1 – Input**

A lung histopathology image is provided to the model.

**Step 2 – CNN Feature Extraction**

GhostNet-100 extracts meaningful local patterns from the image.

**Step 3 – Feature Tokens**

The extracted feature representation is converted into tokens suitable for Transformer processing.

**Step 4 – Self-Attention**

The Transformer analyzes relationships between the feature tokens.

**Step 5 – Feature Enhancement**

The local CNN representation is enhanced with global contextual information.

**Step 6 – Classification**

The final representation is passed through the classification head to predict one of the three classes.

---

## 🧪 Training Configuration

The model is trained using:

| Configuration           | Value                         |
| ----------------------- | ----------------------------- |
| Model                   | GhostNet-100 + 1L Transformer |
| Optimizer               | AdamW                         |
| Learning Rate           | `2e-4`                        |
| Weight Decay            | `1e-2`                        |
| Scheduler               | Cosine Annealing              |
| Maximum Epochs          | 12                            |
| Early Stopping Patience | 3                             |
| Loss Function           | Cross-Entropy Loss            |
| Number of Classes       | 3                             |

### Optimizer

**AdamW** is used for stable optimization and weight-decay-based regularization.

```text
Learning Rate = 0.0002
Weight Decay  = 0.01
```

### Learning Rate Scheduler

A **Cosine Annealing scheduler** gradually reduces the learning rate during training, allowing larger updates in the early stages and smaller updates later in training.

### Early Stopping

Early stopping is used to prevent unnecessary training when validation performance stops improving.

---

## 📊 Model Evaluation

The models are evaluated using standard classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**
* **Training and Validation Loss**

The main comparison is between:

```text
GhostNet-100
       VS
GhostNet-100 + Lightweight Transformer
```

This allows the effectiveness of the proposed Transformer enhancement to be evaluated experimentally.

---

## 🔬 Experimental Comparison

| Model                                    | Main Contribution                             |
| ---------------------------------------- | --------------------------------------------- |
| **GhostNet-100**                         | Lightweight CNN baseline                      |
| **GhostNet-100 + CBAM**                  | Attention-based feature refinement            |
| **GhostNet-100 + 1L Transformer**        | Global relationship modeling                  |
| **GhostNet-100 + CBAM + 1L Transformer** | Combines feature attention and global context |

The final model should be selected based on experimental performance, considering not only accuracy but also **F1-score, generalization, parameter count, and computational efficiency**.

---

## 💡 Key Contribution

The main improvement in this project is the integration of a **lightweight Transformer with GhostNet-100**.

Instead of relying only on CNN-based local feature extraction:

```text
GhostNet-100
     ↓
Local Features
```

the proposed architecture adds:

```text
GhostNet-100
     ↓
Local Features
     +
Transformer
     ↓
Global Relationships
```

This creates a **CNN-Transformer hybrid architecture** designed to capture both local microscopic patterns and broader contextual relationships within lung histopathology images.

---

## 📌 Expected Benefit

The proposed architecture is designed to provide:

* Better representation of complex tissue patterns
* Improved modeling of relationships between image regions
* Efficient feature extraction using GhostNet-100
* Global contextual understanding using self-attention
* A lightweight alternative to using a large Vision Transformer

> **The effectiveness of the proposed model is determined through experimental comparison with the GhostNet-100 baseline.**

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Deep Learning
* Convolutional Neural Networks
* GhostNet-100
* Transformer / Self-Attention
* Computer Vision
* Image Classification
* NumPy
* Matplotlib
* scikit-learn
* Kaggle

---

## 📁 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Data Augmentation
   ↓
GhostNet-100 Baseline
   ↓
Baseline Evaluation
   ↓
Model Enhancement
   ↓
GhostNet-100 + Lightweight Transformer
   ↓
Training
   ↓
Validation
   ↓
Test Evaluation
   ↓
Performance Comparison
```

---

## 🏁 Conclusion

This project investigates a lightweight **CNN-Transformer hybrid architecture** for lung histopathology image classification.

The **GhostNet-100 baseline** provides efficient local feature extraction, while the added **one-layer Lightweight Transformer** introduces self-attention to model relationships between different feature regions.

The overall approach aims to achieve a balance between:

**Accuracy + Feature Representation + Computational Efficiency**

The final performance is validated experimentally against the baseline model using standard classification metrics.
