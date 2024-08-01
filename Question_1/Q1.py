import numpy as np
import pickle

# Hàm softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Trừ max để tăng độ ổn định số học
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Hàm tính toán mất mát (cross-entropy loss)
def compute_loss(y, y_hat):
    m = y.shape[0]
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
    return loss

# Hàm thêm bias (cột 1) vào ma trận đầu vào
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Hàm dự đoán xác suất
def predict_proba(X, W):
    z = np.dot(X, W)
    return softmax(z)

# Hàm dự đoán lớp
def predict(X, W):
    proba = predict_proba(X, W)
    return np.argmax(proba, axis=1)

# Hàm tính accuracy
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Hàm cập nhật trọng số sử dụng gradient descent
def gradient_descent(X, y, W, alpha, iterations):
    m = X.shape[0]
    for i in range(iterations):
        y_hat = predict_proba(X, W)
        loss = compute_loss(y, y_hat)
        gradient = np.dot(X.T, (y_hat - y)) / m
        W -= alpha * gradient
        
        if i % 100 == 0:
            predictions = predict(X, W)
            accuracy = compute_accuracy(np.argmax(y, axis=1), predictions)
            print(f'Iteration {i}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
    
    return W

# Hàm chuyển đổi nhãn thành one-hot encoding
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# Hàm tải dữ liệu từ file CSV
def load_data(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    X = data[:, 1:]  # Các cột đặc trưng
    y = data[:, 0].astype(int)  # Cột nhãn
    return X, y

# Chuẩn bị dữ liệu
X_train, y_train = load_data('Question_3/mnist_train.csv')
X_test, y_test = load_data('Question_3/mnist_test.csv')

# Chuẩn hóa dữ liệu
X_train = X_train / 255.0
X_test = X_test / 255.0

# Thêm bias vào dữ liệu
X_train_bias = add_bias(X_train)
X_test_bias = add_bias(X_test)

# Chuyển đổi nhãn thành one-hot encoding
num_classes = 10
y_train_one_hot = one_hot_encode(y_train, num_classes)
y_test_one_hot = one_hot_encode(y_test, num_classes)

# Khởi tạo trọng số
W = np.random.randn(X_train_bias.shape[1], num_classes) * 0.01

# Huấn luyện mô hình
alpha = 0.01
iterations = 1000
W = gradient_descent(X_train_bias, y_train_one_hot, W, alpha, iterations)

# Lưu mô hình bằng pickle
with open('softmax_regression_model.pkl', 'wb') as f:
    pickle.dump(W, f)

# Tải mô hình từ file
with open('softmax_regression_model.pkl', 'rb') as f:
    W_loaded = pickle.load(f)

# Dự đoán trên tập kiểm tra
y_pred = predict(X_test_bias, W_loaded)
accuracy = compute_accuracy(y_test, y_pred)

print(f'Final Test Accuracy: {accuracy * 100:.2f}%')
