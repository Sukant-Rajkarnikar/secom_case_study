# import pandas as pd
# import numpy as np
# from ucimlrepo import fetch_ucirepo

# def load_data():
#     print("Downloading SECOM dataset...")
#     secom = fetch_ucirepo(id=179)
    
#     x = secom.data.features
#     y = secom.data.targets
    
#     print(f"Features shape: {x.shape}")
#     print(f"Target shape: {y.shape}")
    
#     return x, y

# if __name__ == "__main__":
#     x, y = load_data()


import pandas as pd

def load_and_describe_secom():
    print("Downloading SECOM dataset via pandas...")
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
    
    X = pd.read_csv(base + "secom.data", sep=" ", header=None)
    labels = pd.read_csv(
        base + "secom_labels.data", sep=" ", header=None,
        names=["target", "timestamp"]
    )
    
    y = labels["target"]
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]
    
    print(f"\nTotal wafers (rows): {len(X)}")
    print(f"Total sensors (columns): {len(X.columns)}")

if __name__ == "__main__":
    load_and_describe_secom()