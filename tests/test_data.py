import numpy as np
import pytest
from data_preprocessing import scaler  # Assuming scaler is part of your preprocessing

def test_data_shape():
    X = np.load('X.npy')
    y = np.load('y.npy')
    assert X.shape[0] == y.shape[0], "Mismatch between number of samples in X and y"
    assert X.shape[1] == 60, "Incorrect number of features (should be 60)"
    assert X.shape[2] == 1, "Incorrect feature dimension (should be 1)"

def test_scaling():
    sample_data = np.array([[100], [105], [110]])
    scaled_data = scaler.transform(sample_data)
    assert scaled_data[0] < scaled_data[1] < scaled_data[2], "Scaler didn't work as expected"
