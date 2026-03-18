import numpy as np

class MLP:
    def __init__(self, input_size=30, hidden_size=24, output_size=2):
        # Weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        
        # Biases (initialized to zeros)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))