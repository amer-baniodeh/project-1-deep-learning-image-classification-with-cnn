
# Project-1 - Image Classification CNN
##  with Transfer Learning using MobileNetV2, Image Resizing to 96x96, Data Augmentation and Image upscaling

## About

A high-performance image classification pipeline that solves the classic object detection and categorization problem using deep convolutional neural networks. This project utilizes deep transfer learning with a pre-trained **MobileNetV2** base architecture combined with customized image upscaling, data augmentation pipelines, and progressive classification head tuning to perform multi-class recognition on small-resolution imagery.

- ## Sample Predictions

<center>
<table>
  <td><img src="/images/Horsepred.jpg" height="420" alt="Horse prediction"></td>
  <td> &nbsp &nbsp </td>
  <td><img src="/images/Dookie.jpg" height="420" alt="Dookie prediction"></td>
</table>
</center>

<br>

- ## **Deployment URL:** 

  - the best performing model was deployed on Hugging Face Spaces using a Docker template for Streamlit
  - and can be tested here:🚀 https://huggingface.co/spaces/AntonioTrx99/project1_streamlit

<img src="/images/HFspacesStreamlit.jpg" height="520" alt="Deployed on HFspaces Streamlit">

<br>

---

## Problem Statement

The goal of this project is to build an accurate deep learning classifier capable of correctly categorizing low-resolution ($32 \times 32 \times 3$) objects into one of ten unique real-world classes. Low-resolution images present a significant challenge for modern deep state-of-the-art architectures (like MobileNetV2) because spatial dimensions shrink too quickly across convolutional layers. This project overcomes that structural limitation by introducing a robust preprocessing pipeline featuring bilinear interpolation upscaling, synthetic data generation, and partial layer unfreezing.

<br>


## MODEL TRACKING SPREADSHEET
| Model | Architecture | Learning Rate | Train Time | Accuracy | Notes |
| --- | --- | --- | --- | --- | --- |
| Model 1 | 2 Dense layers | 0.01 | 5 min | 46% | Baseline model - optimizer=SGD |
| Model 2 | 4 conv layers (2 Conv + 2 Dense) | 0.01 | 41s GPU | 58% | VGG |
| Model 3 | 3 layers (1 Conv + 2 Dense configuration) | 0.01 | 1 min | 53% | ONE CHANGE from baseline: Extract spatial features before flattening -> added 1 conv layer |
| Model 4 | 4 layers (2 Conv + 2 Dense) | 0.01 | 1 min GPU | 14% | VGG + Data Augmentation -> brightness & contrast |
| Model 5 | 7 layers | 1.00E-03 | 4 min GPU | 32,5% | Transfer Learning with MobileNetV2 no resize images (32,32) |
| Model 6 | 7 layers | 1.00E-03 | 6 min GPU | 90,1% | Transfer Learning with MobileNetV2 resize images (96,96) |
| Model 7 | 4 layers (2 Conv + 2 Dense)+2 dropout | 0.01 | 2 mins GPU | 56% | Model 2 + 100 Epoch & patience=5 & restore_best_weights=True and 2 dropout (0.25 & 0.5) |
| Model 8 | 13 layers (4 x Conv2D + 4 x BatchNormalization + 2 x MaxPooling2D + 2 x Dropout + 1 x Flatten +2 x Dense) | 0.01 | 1 min GPU | 78% | Optimiser = Adam & more layers |
| Model 9 | 8 layers | 1.00E-05 | 19 min GPU | 94,5% | TL MNV2 with Data Augmentation + Upscaling |
| Model 10 | 7 layers | 0.01-1.00E-06 | 15 min GPU | 90,9% | TL with EfficientNetB0 + Data Aug + Upscaling, Fine Tuning + Adv Learning Rate Scheduler, change to Flatten, higher Dropout and Wider Dense 256 |



<br>



---

## Dataset

* **Source:** CIFAR-10 Dataset (Loaded and prepared via custom numpy splits).
* **Size:** 60,000 total color images.
* **Training Set:** 50,000 samples
* **Validation/Test Set:** 10,000 samples


* **Dimensions:** Original size is $32 \times 32$ pixels with 3 RGB color channels.
* **Classes (10):** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
* **License:** MIT License / Public Domain.

---

## Model Architecture

The core architecture follows a custom **Functional API Pipeline** wrapped around a modified **MobileNetV2** backbone trained originally on ImageNet data.

### Structural Flow:

1. **Input Stage:** Accepts raw $32 \times 32 \times 3$ image matrices.
2. **Data Augmentation Layer:** A `Sequential` block consisting of `RandomFlip("horizontal")`, `RandomRotation(0.05)`, and `RandomZoom(0.05)` layers to prevent over-fitting.
3. **Upscaling Stage:** Uses a Keras `Resizing` layer to scale the input from $32 \times 32$ up to $96 \times 96$ using bilinear interpolation, allowing MobileNetV2 to capture meaningful features.
4. **Feature Extraction Base:** MobileNetV2 (with `weights='imagenet'`, `include_top=False`).
* *Freezing Strategy:* All initial layers are frozen, except for the last 60 layers which are left trainable to adjust features specifically for this domain.


5. **Classification Head:** * `GlobalAveragePooling2D` layer to flatten spatial profiles.
* `Dropout(0.3)` layer to reduce model variance.
* Fully-Connected `Dense(128, activation='relu')` feature layer.
* Output `Dense(10, activation='softmax')` probability layer.



```
[Input: 32x32x3] ──> [Data Augmentation] ──> [Resizing: 96x96x3] ──> [MobileNetV2 Base] ──> [Global Avg Pooling] ──> [Dropout: 0.3] ──> [Dense: 128] ──> [Softmax Output: 10]

```

---

## Results & Performance

The training strategy was executed in two successive blocks (Classification Head Training followed by Targeted Fine-Tuning):

* **Phase 1 (Head Training):** 25 Epochs with `Adam(learning_rate=1e-5)`.
* **Phase 2 (Fine-Tuning):** 10 Epochs with top base layers unfrozen (last 30 layers) and a low learning rate `Adam(learning_rate=1e-5)` to keep pre-trained weights stable.
* **Total Training Runtime:** ~19 minutes on an NVIDIA T4 GPU shape.

### Metrics Table

| Phase | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| --- | --- | --- | --- | --- |
| Final Epoch | **0.1558** | **94.52%** | **0.2790** | **91.02%** |

### Performance Visualizations

<center>
<table>
  <td><img src="/images/Loss_Accuracy_Graphs.png" height="420" alt="Loss_Accuracy_Graphs"></td>
  <td><img src="/images/CNN_Confusion_matrix.png" height="420" alt="CNN_Confusion_matrix"></td>
</table>
</center>
---

## Setup & Installation

Follow these steps to reproduce the training pipeline locally or on an alternative remote cloud instance:

### 1. Clone the repository

```bash
git clone https://github.com/amer-baniodeh/project-1-deep-learning-image-classification-with-cnn
cd cifar10-mobilenetv2-transfer-learning

```

### 2. Environment Setup & Dependency Installation

It is highly recommended to use a Python virtual environment (Python 3.10+):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Executing the Notebook

Open the file inside Jupyter or Google Colab:

```bash
jupyter notebook model_9.ipynb

```

---

## Project Structure

```directory
├── model_9.ipynb            # Core Jupyter Notebook featuring the entire ML pipeline
├── requirements.txt         # Project dependencies and explicit package versions
├── README.md                # Project documentation and performance summary
├── images/                  # Stored confusion matrices, performance plots, and prediction samples
│   ├── confusion_matrix.png
│   ├── curves.png
│   └── sample_predictions.png
└── saved_models/            # Destination for serialized checkpoints (.keras output)
    └── model_9.keras

```

---

## Tech Stack

* **Core Framework:** Python 3.10+
* **Deep Learning Framework:** TensorFlow 2.x & Keras
* **Scientific Computing & Visualization:** NumPy, Matplotlib
* **Evaluation Metrics:** Scikit-Learn

---

## Future Improvements

* **Hyperparameter Optimization:** Conduct Grid/Random search optimizations on dropout frequencies ($0.2$ to $0.5$) and hidden nodes inside the structural dense layers.
* **Learning Rate Schedulers:** Transition from static learning updates to cosine annealing decay strategies (`CosineDecay`) to safely navigate local loss minima.
* **Advanced Architectures:** Evaluate alternative state-of-the-art backbones such as **EfficientNetB0 ( tried and got 90%)** or ConvNeXt to further push classification benchmarks past $95\%$.

---

## Author & Contact

* **Authors Names:** 
    
    - Amer Baniodeh - [github.com/amer-baniodeh](https://github.com/amer-baniodeh)

    - Antonio Traquinas - [github.com/wtraquinas](https://github.com/wtraquinas)

