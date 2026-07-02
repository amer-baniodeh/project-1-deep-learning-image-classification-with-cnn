# project-1-deep-learning 
## image-classification-with-cnn

---

latest model can be tested here:🚀 https://huggingface.co/spaces/AntonioTrx99/project1_streamlit

---

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



**CSV to Markdown Converter** (https://formatarc.com/en/csv-to-markdown/)

<br>

---

---

## Day 1

<br>

\* The **VGG (Visual Geometry Group)** model is a classic, highly influential Convolutional Neural Network (CNN) architecture known for its elegant simplicity, uniform layer structure, and maximum depth. Developed by Oxford University researchers, it is widely used as a baseline for image classification and feature extraction.

Key Architectural Principles:
- Consistent 3x3 Filters: Unlike older models, VGG exclusively uses small 3 × 3 convolutional filters (with a stride of 1). Stacking three 3 × 3 layers has the same effective receptive field as a single large 7 × 7 layer, but introduces more non-linearities and reduces parameters.
- Modular Block Structure: The network is built of blocks. Each block consists of two or three convolutional layers followed by a 2 × 2 max-pooling layer to reduce spatial dimensions.
- Increasing Depth: The number of feature map filters doubles after each pooling operation, scaling systematically from 64 in the early layers up to 512 in the deeper layers.
- Classification Head: The architecture concludes with three Fully Connected (FC) layers: two with 4,096 channels and a final output layer using a Softmax activation to predict class probabilities (e.g., 1,000 for ImageNet).

https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

<br>

---

---

## Day 2

<br>

model 4
- Data Augmentation

model 5
- Transfer Learning

<br>

---

---

<br>

---

---



