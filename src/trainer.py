import numpy as np
from layer import DenseLayer

class MLP:
    def __init__(self, input_dim=30, hidden_dim=24, output_dim=2, batch_size=8):
        self.batch_size = batch_size

        self.layers = [
            DenseLayer(input_dim, hidden_dim),  # Layer 1
            DenseLayer(hidden_dim, hidden_dim), # Layer 2
            DenseLayer(hidden_dim, output_dim)  # Output Layer
        ]

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.5):
        self.batch_size = X_train.shape[0]
        stats = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # 1. Forward Pass
            predictions = self.feedforward(X_train)

            # 2. Compute Loss (Binary Cross-Entropy)
            loss = self.compute_loss(predictions, y_train)
            acc = self.compute_accuracy(predictions, y_train)

            # 3. Backward Pass (Updates weights inside)
            self.backward(X_train, y_train, learning_rate)

            # 4. Validation (Check how we do on unseen data)
            val_preds = self.feedforward(X_val)
            val_loss = self.compute_loss(val_preds, y_val)
            val_acc = self.compute_accuracy(val_preds, y_val)

            # Store stats for plotting later
            stats["train_loss"].append(loss)
            stats["val_loss"].append(val_loss)
            stats["val_acc"].append(val_acc)
            stats["train_acc"].append(acc)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f} - Acc {acc:.2%}")
            
        return stats

    def feedforward(self, X):
        activation = X
        for i, layer in enumerate(self.layers):
            # Pass through weights/bias
            z = layer.forward(activation)
            
            # Apply activation function
            if i == len(self.layers) - 1:
                activation = self.softmax(z) # Last layer
            else:
                activation = self.sigmoid(z) # Hidden layers
                
            # Store these in the layer object for backprop!
            layer.output = activation 
        return activation

    def backward(self, X, Y_true, learning_rate):
        # 1. Handle the Output Layer (Softmax + Cross-Entropy)
        output_layer = self.layers[-1]
        prev_layer_output = self.layers[-2].output

        # Simple subtraction for Softmax + CE
        dZ = output_layer.output - Y_true 

        # Calculate gradients for the last layer
        dW = (prev_layer_output.T @ dZ) / self.batch_size
        db = np.mean(dZ, axis=0, keepdims=True)

        # The error signal to pass backwards
        dA_prev = dZ @ output_layer.weights.T

        # Update the output layer weights immediately or store them
        output_layer.update(dW, db, learning_rate)

        # 2. Loop through Hidden Layers in reverse (starting from the second to last)
        # This is where the 'Logic for a Hidden Layer' lives!
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]

            # Get the input to THIS layer (if it's layer 0, the input is X)
            input_to_this_layer = self.layers[i-1].output if i > 0 else X

            # --- THIS IS THE LOGIC YOU WERE LOOKING FOR ---
            # dZ for hidden layer = Error from next layer * sigmoid derivative
            dZ_hidden = dA_prev * (layer.output * (1 - layer.output))

            # Calculate gradients
            dW_hidden = (input_to_this_layer.T @ dZ_hidden) / self.batch_size
            db_hidden = np.mean(dZ_hidden, axis=0, keepdims=True)

            # Update error signal for the next iteration (moving towards input)
            dA_prev = dZ_hidden @ layer.weights.T

            # Update hidden layer weights
            layer.update(dW_hidden, db_hidden, learning_rate)

    def softmax(self, Z):
        # Subtract max for numerical stability (prevents np.exp from exploding)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def compute_accuracy(self, predictions, y_true):
        # np.argmax gets the index of the highest value (0 or 1)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(pred_labels == true_labels)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, predictions, y_true):
        # Clip predictions to prevent log(0) which is undefined
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        # The formula: -1/N * sum(Y * log(P))
        return -np.mean(np.sum(y_true * np.log(predictions), axis=1))
    
    def save_model(self, scaler, filename="./model/model_weights.npy"):
        checkpoint = {
            # Extract weights and biases from each layer
            "layers": [
                {"w": layer.weights, "b": layer.biases} 
                for layer in self.layers
            ],
            # Extract the normalization parameters
            "scaler": {
                "min": scaler.min,
                "max": scaler.max
            }
        }
        np.save(filename, checkpoint)
        print(f"Checkpoint saved to {filename} ✅")