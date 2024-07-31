import numpy as np

def triplet_loss(anchor, positive, negative, alpha=0.2):
    """
    Compute the Triplet Loss.
    anchor: numpy array, feature vector of the anchor point
    positive: numpy array, feature vector of the positive point
    negative: numpy array, feature vector of the negative point
    alpha: float, margin to ensure separation
    """
    # Compute Euclidean distance
    pos_distance = np.sum((anchor - positive) ** 2)
    neg_distance = np.sum((anchor - negative) ** 2)
    
    # Compute triplet loss
    loss = np.maximum(pos_distance - neg_distance + alpha, 0)
    return loss

# Example usage
anchor = np.array([1.0, 2.0, 3.0])
positive = np.array([1.1, 2.1, 3.1])
negative = np.array([4.0, 5.0, 6.0])

loss = triplet_loss(anchor, positive, negative)
print(f"Triplet Loss: {loss}")
