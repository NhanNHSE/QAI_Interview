import numpy as np

# Preprocessing
class PreProc:
    def __init__(self):
        pass
    
    def one_hot_encode(self, y, levels):
        res = np.zeros((len(y), levels))
        for i in range(len(y)):
            res[i, y[i]] = 1
        return res

    def normalize(self, x):
        return x / np.max(x)

    def read_csv(self, fname):
        data = np.loadtxt(fname, skiprows=1, delimiter=',')
        y = data[:, :1]
        x = data[:, 1:]
        return x, y

    def load_data(self, fname):
        x, y = self.read_csv(fname)
        x = self.normalize(x)
        y = np.int16(y)
        y = self.one_hot_encode(y, levels=10)

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        return x, y

x_train, y_train = PreProc().load_data('mnist_train.csv')
x_test, y_test = PreProc().load_data('mnist_test.csv')

# Neural Network with Dropout and Triplet Loss
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

    def drelu(self, x):
        return (x > 0).astype(float)

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
        self.training = True
        self.x = x.T
        self.z1 = np.dot(self.w1, self.x) + self.b1
        self.a1 = self.relu(self.z1)
        self.a1 = self.dropout(self.a1)

        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.relu(self.z2)
        self.a2 = self.dropout(self.a2)

        self.z3 = np.dot(self.w3, self.a2) + self.b3
        self.out = self.soft_max(self.z3)
        return self.out.T

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        anchor_out = self.forward(anchor)
        positive_out = self.forward(positive)
        negative_out = self.forward(negative)

        pos_dist = np.sum((anchor_out - positive_out) ** 2, axis=1)
        neg_dist = np.sum((anchor_out - negative_out) ** 2, axis=1)

        loss = np.mean(np.maximum(0, pos_dist - neg_dist + margin))
        return loss

    def accuracy(self, x, y):
        predictions = np.argmax(self.forward(x), axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

    def backward(self, anchor, positive, negative):
        # Placeholder function. Implement gradient calculation and update
        pass

def adjust_learning_rate(lr, epoch, decay_rate=0.95):
    return lr * (decay_rate ** epoch)

epochs = 20
batch_size = 1000
shuffle = True
lr = 1e-3

def shuffle_data(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]

x_train, y_train = PreProc().load_data('mnist_train.csv')
x_test, y_test = PreProc().load_data('mnist_test.csv')

nn = NeuralNetwork(x_train.shape[1], 256, 128, y_train.shape[1], lr=lr, dropout_rate=0.5)

l = []
acc = []

for epoch in range(epochs):
    loss = 0
    accuracy = 0
    if shuffle:
        x_train, y_train = shuffle_data(x_train, y_train)
    for batch in range(x_train.shape[0] // batch_size):
        x = x_train[batch*batch_size: (batch+1)*batch_size]
        split_size = batch_size // 3
        anchor = x[:split_size]
        positive = x[split_size:2*split_size]
        negative = x[2*split_size:]

        loss += nn.triplet_loss(anchor, positive, negative)
        nn.backward(anchor, positive, negative)

    loss = loss / (x_train.shape[0] // batch_size)
    accuracy = nn.accuracy(x_test, y_test)
    l.append(loss)
    acc.append(accuracy)
    print(f'Epoch {epoch}: loss = {loss}, accuracy = {accuracy}')

    # Adjust learning rate
    lr = adjust_learning_rate(lr, epoch)
    nn.lr = lr
