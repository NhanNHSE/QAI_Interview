import numpy as np

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

class NeuralNetwork:
    def __init__(self, d_in, d1, d2, d_out, lr=1e-3):
        self.d_in = d_in
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out
        self.lr = lr
        self.init_weights()
        
    def init_weights(self):
        self.w1 = np.random.randn(self.d1, self.d_in)
        self.b1 = np.random.randn(self.d1, 1)
        
        self.w2 = np.random.randn(self.d2, self.d1)
        self.b2 = np.random.randn(self.d2, 1)
        
        self.w3 = np.random.randn(self.d_out, self.d2)
        self.b3 = np.random.randn(self.d_out, 1)

    def relu(self, x):
        return np.maximum(x, 0)

    def drelu(self, x):
        return np.diag(1.0 * (x > 0))

    def soft_max(self, x):
        x = x - np.max(x, axis=0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)
  
    def forward(self, x, y):
        self.x = x
        self.y = y
        
        self.z1 = np.matmul(self.w1, self.x) + self.b1
        self.a1 = np.apply_along_axis(self.relu, 1, self.z1)
        
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = np.apply_along_axis(self.relu, 1, self.z2)
        
        self.z3 = np.matmul(self.w3, self.a2) + self.b3
        self.out = np.apply_along_axis(self.soft_max, 1, self.z3)

    def transpose(self, x):
        return np.transpose(x, [0, 2, 1])

    def backward(self):
        delta = 2 * self.transpose(self.out - self.y)
        self.dw3 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.a2)),
            axis=0
        )
        self.db3 = np.mean(self.transpose(delta), axis=0)
        
        delta = np.matmul(
            np.matmul(delta, self.w3),
            np.squeeze(np.apply_along_axis(self.drelu, 1, self.z2))
        )
        self.dw2 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.a1)),
            axis=0
        )
        self.db2 = np.mean(self.transpose(delta), axis=0)
        
        delta = np.matmul(
            np.matmul(delta, self.w2),
            np.squeeze(np.apply_along_axis(self.drelu, 1, self.z1))
        )
        self.dw1 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.x)),
            axis=0
        )
        self.db1 = np.mean(self.transpose(delta), axis=0)
        
        self.w3 -= self.lr * self.dw3
        self.b3 -= self.lr * self.db3
        
        self.w2 -= self.lr * self.dw2
        self.b2 -= self.lr * self.db2
        
        self.w1 -= self.lr * self.dw1
        self.b1 -= self.lr * self.db1

def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]
