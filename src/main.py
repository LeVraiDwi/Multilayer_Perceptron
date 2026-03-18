import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from min_max_scaler import min_max_scaler

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
    data = raw.loc[:, raw.columns != 1]
    train_x, validation_x, train_y, validation_y = train_test_split(data, label, stratify=label, random_state=42, test_size=0.4)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    plt.plot(train_x)
    print(train_x.shape, train_y.shape)

    plt.show()

    clf = Pipeline([
        ('min_max_scale', min_max_scaler()),
    ], verbose=False)
    
    clf.fit(train_x, train_y)

    clf.transform(train_x)
    
    
if __name__ == "__main__":
    main()