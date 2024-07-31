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
def init_params(input_dim, output_dim):
    W = np.random.randn(output_dim, input_dim + 1) * 0.01
    return W

# Add bias term to the input
def pad(x):
    return np.concatenate((np.ones((1, x.shape[1]), dtype=x.dtype), x), axis=0)

# Sigmoid activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Compute Cross-Entropy Loss
def compute_loss(Y_pred, Y):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(Y_pred + 1e-8) + (1 - Y) * np.log(1 - Y_pred + 1e-8)) / m
    return loss

# Forward propagation
def forward_prop(W, X):
    Z = np.dot(W, X)
    A = sigmoid(Z)
    return Z, A

# Backward propagation
def backward_prop(Z, A, Y, X):
    m = X.shape[1]
    dZ = A - Y
    dW = np.dot(dZ, X.T) / m
    return dW

# Update parameters using gradient descent
def update_params(W, dW, alpha):
    W -= alpha * dW
    return W

# Get predictions
def get_predictions(A):
    return (A > 0.5).astype(int)

# Calculate accuracy
def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# Save model parameters
def save_params(W, W_path):
    np.save(W_path, W)

# Load model parameters
def load_params(W_path):
    W = np.load(W_path)
    return W

# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    input_dim = X.shape[0]
    output_dim = 1  # Single output for binary classification
    
    W = init_params(input_dim, output_dim)
    
    X_padded = pad(X)
    
    for i in range(iterations):
        Z, A = forward_prop(W, X_padded)
        dW = backward_prop(Z, A, Y, X_padded)
        W = update_params(W, dW, alpha)
        
        if i % 100 == 0:
            loss = compute_loss(A, Y)
            predictions = get_predictions(A)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")
    
    return W

# Example usage
if __name__ == "__main__":
    # Load the MNIST dataset
    train_images, train_labels = load_mnist_from_csv(r'C:\FPT\SEMESTER_6\PV\Question_1\data\mnist_train.csv')
    test_images, test_labels = load_mnist_from_csv(r'C:\FPT\SEMESTER_6\PV\Question_1\data\mnist_test.csv')

    # Normalize the images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Transpose images to match the expected input shape
    train_images = train_images.T
    test_images = test_images.T

    # Convert labels to binary (for example, classify digit 0 vs. not 0)
    train_labels = (train_labels == 0).astype(int)
    test_labels = (test_labels == 0).astype(int)

    alpha = 0.01
    iterations = 1000

    # Train the model
    W = gradient_descent(train_images, train_labels, alpha, iterations)

    # Save the trained parameters
    save_params(W, 'W.npy')

    # Test the model
    W = load_params('W.npy')
    _, A_test = forward_prop(W, pad(test_images))
    predictions = get_predictions(A_test)
    accuracy = get_accuracy(predictions, test_labels)

    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
