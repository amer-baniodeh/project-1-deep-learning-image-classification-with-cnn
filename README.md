# project-1-deep-learning 
## image-classification-with-cnn


---

<br>

## MODEL TRACKING SPREADSHEET
| Model | Architecture | Learning Rate | Train Time | Accuracy | Notes |
| --- | --- | --- | --- | --- | --- |
| Model 1 | 2 Dense layers | 0.01 | 5 min | 46% | Baseline - start here |
| Model 2 | 4 conv layers (2 Conv + 2 Dense) | 0.01 | 41s GPU | 58% | VGG* |
| Model 3 | 3 layers (1 Conv + 2 Dense configuration) | 0.01 | x min | 53% | ONE CHANGE from baseline: Extract spatial features before flattening -> added 1 conv layer |
| --- | --- | --- | --- | --- | --- |


**CSV to Markdown Converter** (https://formatarc.com/en/csv-to-markdown/)



---

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
