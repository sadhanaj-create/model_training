import torch
from model import LSTMStockPredictor

def test_model_forward():
    model = LSTMStockPredictor()
    # Create a dummy input tensor (batch_size=1, seq_length=60, feature=1)
    X = torch.randn(1, 60, 1)
    output = model(X)
    assert output.shape == (1, 1), "Model output shape should be (1, 1)"
