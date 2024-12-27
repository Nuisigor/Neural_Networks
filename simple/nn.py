import numpy as np
import pandas as pd

def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

class ReLU:
    @staticmethod
    def func(Z):
        return np.maximum(Z, 0)
    
    @staticmethod
    def deriv(Z):
        return Z > 0
    
    @staticmethod
    def errorLastLayer(A, y):
        one_hot_y = oneHot(y)
        return 2 * (A - one_hot_y) * ReLU.deriv(A)
    
class SoftMax:
    @staticmethod
    def func(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    @staticmethod
    def errorLastLayer(A, y):
        one_hot_y = oneHot(y)
        return A - one_hot_y

class NN:
    
    class Layer:
        def __init__(self, dataSize, size, outputSize, activation, last):
            self.size = size
            self.outputSize = outputSize
            self.W = np.random.rand(outputSize, size) - 0.5
            self.b = np.random.rand(outputSize, 1) - 0.5
            self.activation = activation
            self.last = last
            self.dataSize = dataSize
        
        def forward(self, X):
            Z = self.W.dot(X) + self.b
            A = self.activation.func(Z)
            return Z, A
        
        def backward(self, W, dZ, A, A_prev, Z_prev, y):         
            dZ = self.activation.errorLastLayer(A, y) if self.last else W.T.dot(dZ) * self.activation.deriv(Z_prev)           
            dW = 1 / self.dataSize * dZ.dot(A_prev.T)
            db = 1 / self.dataSize * np.sum(dZ)
            return dZ, dW, db
        
        def updateParams(self, dW, db, alpha):
            self.W = self.W - alpha * dW
            self.b = self.b - alpha * db
    
    class Metrics:
        @staticmethod
        def getPredictions(A):
            return np.argmax(A, 0)
        
        @staticmethod
        def getAccuracy(predictions, y):
            return np.sum(predictions == y) / y.size
            
    
    def __init__(self, dataSize):
        self.layers = []
        self.layers_count = 0
        self.dataSize = dataSize
    
    def addLayer(self, size, outputSize, activation):
        if self.layers_count >= 1:
            self.layers[-1].last = False
        self.layers.append(NN.Layer(self.dataSize, size, outputSize, activation, True))
        self.layers_count += 1
    
    def fit(self, X, y, alpha, iterations):
        self.dataSize, m = X.shape
        
        for i in range(iterations):
            A = X
            results = [[0, A]]
            
            for layer in self.layers:
                Z, A = layer.forward(A)
                results.append([Z, A])
                
            if i % 10 == 0:
                predictions = NN.Metrics.getPredictions(A)
                print(NN.Metrics.getAccuracy(predictions, y))
            
            dZ = 0
            for layer in reversed(self.layers):
                Z, A = results.pop()
                A_prev = results[-1][1] if len(results) > 0 else X
                W = self.layers[self.layers.index(layer) + 1].W if self.layers.index(layer) + 1 < len(self.layers) else None
                dZ, dW, db = layer.backward(W, dZ, A, A_prev, Z, y)
                layer.updateParams(dW, db, alpha)
                

## Prepare dataset
data = pd.read_csv('train.csv')
data = np.array(data)
m , n = data.shape
np.random.shuffle(data) 

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

## Train model
nn = NN(m)
nn.addLayer(784, 10, ReLU)
nn.addLayer(10, 10, ReLU)
nn.addLayer(10, 10, SoftMax)

nn.fit(X_train, Y_train, 0.10, 500)