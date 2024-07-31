import numpy as np

def triplet_loss_multi_samples(anchor, positives, negatives, margin=0.2):
    """
    Compute the Triplet Loss for multiple positives and negatives.

    Arguments:
    anchor -- numpy array of shape (m, n), embeddings for the anchor images
    positives -- numpy array of shape (m, k, n), embeddings for the positive images
    negatives -- numpy array of shape (m, l, n), embeddings for the negative images
    alpha -- margin

    Returns:
    loss -- float, the value of the triplet loss.
    """
    # Compute the L2 distance between anchor and all positives, then average
    pos_dist = np.mean(np.sum(np.square(anchor[:, np.newaxis, :] - positives), axis=2), axis=1)

    # Compute the L2 distance between anchor and all negatives, then average
    neg_dist = np.mean(np.sum(np.square(anchor[:, np.newaxis, :] - negatives), axis=2), axis=1)

    # Compute the triplet loss
    loss = np.maximum(0, pos_dist - neg_dist + margin)

    return np.mean(loss)