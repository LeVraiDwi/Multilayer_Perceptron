import pandas as pd

def load_data(path = "./data/data.csv") -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)

    return raw
