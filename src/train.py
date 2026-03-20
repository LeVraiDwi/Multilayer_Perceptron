import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from min_max_scaler import min_max_scaler
from trainer import MLP



def main():
    parser = argparse.ArgumentParser(description="EEG Tool")
    # Create subparsers. required=True means they MUST pick 'train' or 'predict'
    parser.add_argument('--dataset', type = str, required=True)
    
    args = parser.parse_args()
    try:
        raw = load_data(args.dataset)
    except:
        print(f'fail to load the file: {args.dataset}')
        return
    
    label = raw[1]
    data = raw.loc[:, 2:]
    
    train_x, temp_x, train_y, temp_y = train_test_split(data, label, stratify=label, random_state=42, test_size=0.4)
    # Second split: Divide the 40% "Temp" into 50/50 (which is 20% / 20% of the total)
    validation_x, test_x, validation_y, test_y = train_test_split(
        temp_x, temp_y, stratify=temp_y, random_state=42, test_size=0.5
    )

    # Combine features and labels into one DataFrame
    test_dataset = pd.concat([test_y, test_x], axis=1)

    # Save to CSV (without the index to keep it clean)
    test_dataset.to_csv("./data/test_data.csv", index=False, header=False)
    print("Test data saved to test_data.csv 📝")
    
    # 1. Prepare Features
    scaler = min_max_scaler()
    scaler.fit(train_x, train_y) # This also computes your one-hot labels

    X_train_scaled = scaler.transform(train_x)
    X_val_scaled = scaler.transform(validation_x)
    
    # 2. Get One-Hot Labels (assuming your scaler handles this as in your previous snippet)
    Y_train_oh = scaler.y_one_hot 
    # You'll need a way to get one-hot for validation too!
    Y_val_oh = np.column_stack((validation_y == 'B', validation_y == 'M')).astype(int)

    # 3. Initialize and Train
    model = MLP(input_dim=30, hidden_dim=24, output_dim=2, batch_size=len(X_train_scaled))
    
    # Your fit method now gets everything it needs
    history = model.fit(
        X_train_scaled, Y_train_oh, 
        X_val_scaled, Y_val_oh, 
        epochs=1000, learning_rate=0.1
    )

    model.save_model(scaler, "./model/model.npy")

    # 4. Plotting (Part of the project requirement!)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.show()
    
    plt.plot(history['train_acc'], label='Train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()