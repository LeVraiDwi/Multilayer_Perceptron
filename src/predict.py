import numpy as np
import pandas as pd
from trainer import MLP
from load_data import load_data


def load_and_predict(checkpoint_path, data_to_test):
    # 1. Load the bundled dictionary
    checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
    
    # Load the data
    try:
        raw = load_data("data/test_data.csv")
    except:
        print(f'fail to load the file: data/test_data.csv')
        return

    # Separate labels (column 0) and features (columns 1 to the end)
    label = raw[0]

    y_one_hot = np.column_stack((label == 'B', label == 'M')).astype(int)
    data = raw.loc[:, 1:].to_numpy().astype(float)

    # 2. Setup the model (dimensions must match your training!)
    model = MLP(input_dim=30, hidden_dim=24, output_dim=2)
    
    # 3. Assign weights to layers
    for i, layer in enumerate(model.layers):
        layer.weights = checkpoint["layers"][i]["w"]
        layer.biases = checkpoint["layers"][i]["b"]
    
    
    # 4. Scale the input using the bundled scaler params
    s_min = np.array(checkpoint["scaler"]["min"]).flatten()
    s_max = np.array(checkpoint["scaler"]["max"]).flatten()
    
    X_scaled = (data - s_min) / (s_max - s_min + 1e-8)
    print(f"Data columns: {data.shape[1]}")
    print(f"s_min shape: {s_min.shape}")
    print(f"First 5 values of data[0]: {data[0][:5]}")
    print(f"Data shape: {data.shape}")
    print(f"Scaler min shape: {s_min.shape}")
    
    # 5. Get probabilities and final class
    probs = model.feedforward(X_scaled)
    predictions = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_one_hot, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}% 🏆")
    return predictions

def main():
    pred = load_and_predict("model/model.npy", "data/test_data.csv")


if __name__ == "__main__":
    main()