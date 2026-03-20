import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Weights matrix: (inputs, neurons)
        # We initialize with small random values
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        
        # Bias vector: (1, neurons)
        # Usually initialized to zero
        self.biases = np.zeros((1, n_neurons))
        
        # We will need to store these during forward pass for backprop later
        self.input = None
        self.z = None
        self.output = None

    def forward(self, inputs):
        self.input = inputs
        # Z = X @ W + b
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.z
    
    def update(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db