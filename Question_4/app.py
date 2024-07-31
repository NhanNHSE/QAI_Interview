import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

class PreProc:
    def one_hot_encode(self, y, levels):
        res = np.zeros((len(y), levels))
        for i in range(len(y)):
            res[i, y[i]] = 1
        return res

    def normalize(self, x):
        return x / 255.0  # MNIST pixel values are from 0 to 255

    def preprocess_image(self, image):
        image = image.resize((28, 28)).convert('L')  # MNIST images are 28x28 grayscale
        image = np.array(image)
        image = self.normalize(image)
        image = image.flatten()  # Flatten to 1D array
        return np.expand_dims(image, axis=0)  # Add batch dimension

class NeuralNetwork:
    def __init__(self, d_in, d1, d2, d_out, lr=1e-3, dropout_rate=0.5):
        self.d_in = d_in
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.init_weights()

    def init_weights(self):
        self.w1 = np.random.randn(self.d1, self.d_in)
        self.b1 = np.zeros((self.d1, 1))
        self.w2 = np.random.randn(self.d2, self.d1)
        self.b2 = np.zeros((self.d2, 1))
        self.w3 = np.random.randn(self.d_out, self.d2)
        self.b3 = np.zeros((self.d_out, 1))

    def relu(self, x):
        return np.maximum(x, 0)

    def dropout(self, x):
        if self.training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            return x * mask
        return x

    def soft_max(self, x):
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, x):
        self.training = False
        self.x = x.T
        self.z1 = np.dot(self.w1, self.x) + self.b1
        self.a1 = self.relu(self.z1)
        self.a1 = self.dropout(self.a1)

        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.relu(self.z2)
        self.a2 = self.dropout(self.a2)

        self.z3 = np.dot(self.w3, self.a2) + self.b3
        self.out = self.soft_max(self.z3)
        return self.out.T  # Ensure output shape is (batch_size, d_out)

    def predict(self, x):
        out = self.forward(x)
        return np.argmax(out, axis=1)[0]  # Ensure return is a single value

# Initialize PreProc and NeuralNetwork instances
preproc = PreProc()
nn = NeuralNetwork(d_in=784, d1=256, d2=128, d_out=10)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            image = preproc.preprocess_image(image)
            print(f'Preprocessed image shape: {image.shape}')
            
            prediction = nn.predict(image)
            print(f'Prediction: {prediction}')
            
            return render_template('index.html', prediction=int(prediction))
        except Exception as e:
            print(f'Error during prediction: {e}')
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
