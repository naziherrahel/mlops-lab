import pandas as pd
from sklearn.datasets import load_iris

def normalize(data):
    return (data - data.mean()) / data.std()

def load_and_normalize_iris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return normalize(df)

if __name__ == "__main__":
    df_norm = load_and_normalize_iris()
    print(df_norm.head())
