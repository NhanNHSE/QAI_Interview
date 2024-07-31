from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model parameters
with open('nn_model.pkl', 'rb') as f:
    W = pickle.load(f)  # Loaded model should be a numpy array

def preprocess_image(image_file):
    try:
        img = Image.open(image_file).convert('L')
        img = img.resize((28, 28))  # Assuming the model expects 28x28 images
        img_array = np.array(img).reshape(-1)
        img_array = img_array / 255.0
        return img_array.reshape(-1, 1)  # Reshape for compatibility with model input
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # To ensure numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def predict(W, X):
    X_padded = np.vstack([np.ones((1, X.shape[1])), X])  # Add bias term
    Z = np.dot(W, X_padded)
    A = softmax(Z)
    return A

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img_array = preprocess_image(file)
    if isinstance(img_array, tuple):  # In case of error during preprocessing
        return img_array

    A = predict(W, img_array)
    prediction = np.argmax(A)
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
