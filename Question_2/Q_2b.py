import numpy as np

def extended_triplet_loss(anchor, positives, negatives, alpha=0.2):
    """
    Compute the Triplet Loss with multiple positive and negative samples.
    anchor: numpy array, feature vector of the anchor point
    positives: numpy array, feature vectors of positive points, shape (n_positives, dim)
    negatives: numpy array, feature vectors of negative points, shape (n_negatives, dim)
    alpha: float, margin to ensure separation
    """
    # Compute distances between anchor and all positive points
    pos_distances = np.sum((anchor - positives) ** 2, axis=1)
    
    # Compute distances between anchor and all negative points
    neg_distances = np.sum((anchor - negatives) ** 2, axis=1)
    
    # Compute loss for each positive-negative pair
    loss = 0
    for pos_dist in pos_distances:
        for neg_dist in neg_distances:
            loss += np.maximum(pos_dist - neg_dist + alpha, 0)
    
    return loss

# Example usage
anchor = np.array([1.0, 2.0, 3.0])
positives = np.array([[1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
negatives = np.array([[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]])

loss = extended_triplet_loss(anchor, positives, negatives)
print(f"Extended Triplet Loss: {loss}")
