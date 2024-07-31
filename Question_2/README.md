# Triplet Loss
## 1. Definition
Triplet Loss is a commonly used loss function in image recognition and matching problems, especially in deep learning models. The goal of Triplet Loss is to ensure that examples of the same class are closer to each other in the feature space than examples of different classes.
## 2. Formula Triplet Loss with One Samples
The Triplet Loss can be expressed as:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$

Where:
- $x_i^a$ is the anchor.
- $x_i^p$ is the positive example.
- $x_i^n$ is the negative example.
- $α$ is the margin.
- $N$ is the number of triplets used in the loss calculation.

## 3. Formula Triplet Loss with Multiple Samples
$$\mathcal{L} = \frac{1}{A} \sum_{i=1}^{N} \max\left(0, \frac{1}{P} \sum_{p\in P}\|f(x_i^a) - f(x_i^p)\|^2 - \frac{1}{N}\sum_{n\in N} \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$

Where:
- $x_i^a$ is the anchor.
- $x_i^p$ is the positive example.
- $x_i^n$ is the negative example.
- $α$ is the margin.
- $P$ is number of positive samples.
- $N$ is number of negative samples.
- $A$ is the number of triplets used in the loss calculation.

## 3. Advantages & Disadvantages
### 3.1. Advantages
- Discriminative Feature Learning
- Robust to Class Imbalance
- Good for High-Dimensional Data

### 3.2. Disadvantages
- Margin Sensitivity
- Selection of Triplets
- Computational Cost
## 4. Applications
- **Face Recognition**: Ensures that faces of the same person are closer in the feature space compared to faces of different people.
- **Image Retrieval**: Helps in learning embeddings such that similar images are closer together in the feature space.

## 5. Files
- `triplet_loss_one_sample.py`: Contain Triplet Loss implementation with 1 anchor and 1 fake sample
- `triplet_loss_multi_samples.py`: Contain Triplet Loss implementation with 2 anchors and 5 fake samples
- `notebook/Triplet_Loss.ipynb`: contain example implementation with one and multiple samples
