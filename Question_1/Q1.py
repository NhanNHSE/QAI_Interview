import numpy as np
import pickle  # Thêm import pickle

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

# Softmax activation function
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Compute Cross-Entropy Loss
def compute_loss(Y_pred, Y):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_pred + 1e-8)) / m
    return loss

# Forward propagation
def forward_prop(W, X):
    Z = np.dot(W, X)
    A = softmax(Z)
    return Z, A

# Backward propagation
def backward_prop(A, Y, X):
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
    return np.argmax(A, axis=0)

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

# Save model using pickle
def save_model_pickle(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load model using pickle
def load_model_pickle(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    input_dim = X.shape[0]
    output_dim = Y.shape[0]  # Number of classes
    
    W = init_params(input_dim, output_dim)
    
    X_padded = pad(X)
    
    for i in range(iterations):
        Z, A = forward_prop(W, X_padded)
        dW = backward_prop(A, Y, X_padded)
        W = update_params(W, dW, alpha)
        
        if i % 100 == 0:
            loss = compute_loss(A, Y)
            predictions = get_predictions(A)
            accuracy = get_accuracy(predictions, np.argmax(Y, axis=0))
            print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")
    
    return W

# Convert labels to one-hot encoding
def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# Example usage
if __name__ == "__main__":
    # Load the MNIST dataset
    train_images, train_labels = load_mnist_from_csv('C:\FPT\SEMESTER_6\PV\Question_1\data\mnist_train.csv')
    test_images, test_labels = load_mnist_from_csv('C:\FPT\SEMESTER_6\PV\Question_1\data\mnist_test.csv')

    # Normalize the images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Transpose images to match the expected input shape
    train_images = train_images.T
    test_images = test_images.T

    # Convert labels to one-hot encoding
    num_classes = 10  # 10 classes for MNIST
    train_labels_one_hot = one_hot(train_labels, num_classes)
    test_labels_one_hot = one_hot(test_labels, num_classes)

    alpha = 0.01
    iterations = 1000

    # Train the model
    W = gradient_descent(train_images, train_labels_one_hot, alpha, iterations)

    # Save the trained parameters
    save_params(W, 'W.npy')

    # Save the model using pickle
    save_model_pickle(W, 'nn_model.pkl')

    # Load the model using pickle
    W_loaded = load_model_pickle('nn_model.pkl')

    # Test the model
    _, A_test = forward_prop(W_loaded, pad(test_images))
    predictions = get_predictions(A_test)
    accuracy = get_accuracy(predictions, np.argmax(test_labels_one_hot, axis=0))

    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
