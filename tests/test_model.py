from model import train_model

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy >= 0.85, "Model accuracy below 85%!"
