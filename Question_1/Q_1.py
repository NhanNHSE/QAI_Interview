import numpy as np

# Function to load MNIST data from CSV file using numpy
def load_mnist_from_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    labels = data[:, 0].astype(int)
    images = data[:, 1:]
    return images, labels

# Normalize the images
def normalize(images):
    return images.astype(np.float32) / 255.0

# Initialize parameters
def init_params(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(hidden_dim, input_dim + 1) * 0.5
    W2 = np.random.randn(output_dim, hidden_dim) * 0.5
    return W1, W2

# Add bias term to the input
def pad(x):
    return np.concatenate((np.ones((1, x.shape[1]), dtype=x.dtype), x), axis=0)

# ReLU activation function
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax activation function
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # For numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Forward propagation
def forward_prop(W1, W2, X):
    Z1 = np.dot(W1, X)
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivative of ReLU
def ReLU_deriv(Z):
    return Z > 0

# Convert labels to one-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m_train = X.shape[1]
    one_hot_Y = one_hot(Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = np.dot(dZ2, A1.T) / m_train
    dZ1 = np.dot(W2.T, dZ2) * ReLU_deriv(Z1)
    dW1 = np.dot(dZ1, X.T) / m_train
    
    return dW1, dW2

# Update parameters using gradient descent
def update_params(W1, W2, dW1, dW2, alpha):
    W1 -= alpha * dW1
    W2 -= alpha * dW2
    return W1, W2

# Get predictions
def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Calculate accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Save model parameters
def save_params(W1, W2, W1_path, W2_path):
    np.save(W1_path, W1)
    np.save(W2_path, W2)

# Load model parameters
def load_params(W1_path, W2_path):
    W1 = np.load(W1_path)
    W2 = np.load(W2_path)
    return W1, W2

# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    input_dim = X.shape[0]
    hidden_dim = 128  # Example hidden dimension size
    output_dim = 10   # Number of classes
    
    W1, W2 = init_params(input_dim, hidden_dim, output_dim)
    
    X_padded = pad(X)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, W2, X_padded)
        dW1, dW2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_padded, Y)
        W1, W2 = update_params(W1, W2, dW1, dW2, alpha)
        
        if i % 100 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {accuracy * 100:.2f}%")
    
    return W1, W2

# Example usage
if __name__ == "__main__":
    # Load the MNIST dataset
    train_images, train_labels = load_mnist_from_csv('mnist_train.csv')
    test_images, test_labels = load_mnist_from_csv('mnist_test.csv')

    # Normalize the images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Transpose images to match the expected input shape
    train_images = train_images.T
    test_images = test_images.T

    alpha = 0.01
    iterations = 1000

    # Train the model
    W1, W2 = gradient_descent(train_images, train_labels, alpha, iterations)

    # Save the trained parameters
    save_params(W1, W2, 'W1.npy', 'W2.npy')

    # Test the model
    W1, W2 = load_params('W1.npy', 'W2.npy')
    _, _, _, A2 = forward_prop(W1, W2, pad(test_images))
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, test_labels)

    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
