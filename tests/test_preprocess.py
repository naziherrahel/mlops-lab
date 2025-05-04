from preprocess import load_and_normalize_iris
import numpy as np

def test_preprocessing_output():
    df = load_and_normalize_iris()
    assert not df.isnull().values.any(), "Missing values found after normalization!"
    assert np.allclose(df.mean(), 0, atol=0.1), "Data mean not close to 0!"
