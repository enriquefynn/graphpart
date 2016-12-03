import numpy as np

class Network:
    def __init__(self, layer_sizes, std = 1.0):
        self.layer_sizes = layer_sizes
        self.weights = []
        
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights.append(np.random.normal(0.0, std, (layer_sizes[i+1], layer_sizes[i])))
        self.buffers = [None] * len(self.layer_sizes)
            
    def forward(self, matrix, features):
        n_nodes = features.shape[1]
        n_features = features.shape[0]
        self.buffers[0] = features.copy()

        for i in range(len(self.weights)):
            xx = self.buffers[i]
            
            step = np.dot(self.weights[i], xx)
            xxh = matrix.dot(step.T).T
                        
            if i < len(self.weights) - 1:
                #self.buffers[i+1] = xxh
                self.buffers[i+1] = np.tanh(xxh)
            else:
                self.buffers[i+1] = np.exp(xxh) / np.sum(np.exp(xxh), axis=0, keepdims=True)
        return self.buffers[-1]