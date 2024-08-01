from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pickle
from PIL import Image
import io

app = Flask(__name__)

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Function to predict probabilities
def predict_proba(X, W):
    z = np.dot(X, W)
    return softmax(z)

# Function to predict class labels
def predict(X, W):
    proba = predict_proba(X, W)
    return np.argmax(proba, axis=1)

# Function to add bias (column of 1s) to input matrix
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Function to preprocess image
def preprocess_image(image):
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((28, 28))  # Resize image
    image_array = np.array(image)
    image_array = image_array.flatten()  # Flatten to 1D array
    image_array = image_array / 255.0  # Normalize image
    image_array = add_bias(image_array.reshape(1, -1))  # Add bias
    return image_array

# Load the model from file
with open('softmax_regression_model.pkl', 'rb') as f:
    W = pickle.load(f)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    image_array = preprocess_image(image)
    prediction = predict(image_array, W)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
