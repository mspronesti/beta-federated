import pytest
import torch
from models import CNN


@pytest.mark.parametrize(
    ["input_shape", "in_features", "num_classes"],
    [
        ((2, 1, 28, 28), 1, 10),
    ],
)
def test_cnn_forward(input_shape, in_features, num_classes):
    x = torch.FloatTensor(*input_shape)
    model = CNN(in_features, num_classes)
    logits = model(x)
    assert logits.shape == (input_shape[0], num_classes)
