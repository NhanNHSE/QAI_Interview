import numpy as np

def triplet_loss_one_sample(anchor, positive, negative, margin=1.0):
    """
    Computes the Triplet Loss for multiple anchors, positives, and negatives.

    Parameters:
    - anchor: np.ndarray, feature vector of the anchors.
    - positive: np.ndarray, feature vector of the positive.
    - negative: np.ndarray, feature vector of the negative.
    - margin: float, margin for calculating the loss.

    Returns:
    - total_loss: float, the value of the triplet loss.
    """
    # Tính toán khoảng cách bình phương giữa anchor và positive
    pos_dist = np.sum(np.square(anchor - positive), axis=-1)
    
    # Tính toán khoảng cách bình phương giữa anchor và negative
    neg_dist = np.sum(np.square(anchor - negative), axis=-1)
    
    # Tính toán Triplet Loss
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return loss