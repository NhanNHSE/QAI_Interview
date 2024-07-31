import numpy as np
import pickle
from utils import PreProc, NeuralNetwork, shuffle

def main():
    epochs = 20
    batch_size = 1000
    shuffle_data = True
    lr = 1e-3

    x_train, y_train = PreProc().load_data('Question_3/mnist_train.csv')
    x_test, y_test = PreProc().load_data('Question_3/mnist_test.csv')

    nn = NeuralNetwork(x_train.shape[1], 256, 128, y_train.shape[1], lr=lr)

    l = []
    acc = []

    for i in range(epochs):
        loss = 0
        accuracy = 0
        if shuffle_data:
            x_train, y_train = shuffle(x_train, y_train)
        for batch in range(x_train.shape[0] // batch_size):
            x = x_train[batch * batch_size: (batch + 1) * batch_size]
            y = y_train[batch * batch_size: (batch + 1) * batch_size]
            nn.forward(x, y)
            loss += np.mean((nn.out - nn.y) ** 2)
            accuracy += np.mean(np.argmax(nn.out, axis=1) == np.argmax(nn.y, axis=1))
            nn.backward()
        loss = loss / (x_train.shape[0] // batch_size)
        l.append(loss)
        accuracy = accuracy / (x_train.shape[0] // batch_size)
        acc.append(accuracy)
        print(f'Epoch {i}: loss = {loss}, accuracy = {accuracy}')
    
    filename = 'nn_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(nn, file)

if __name__ == "__main__":
    main()
