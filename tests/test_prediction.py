import numpy as np
import pytest
from PIL import Image

from prediction import normalize_age_prediction, normalize_gender_prediction, preprocess_image


def test_preprocess_image_returns_normalized_rgb_batch():
    image = Image.new("RGBA", (128, 80), (255, 0, 0, 128))

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 64, 64, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_gender_prediction_handles_softmax_like_output():
    label, confidence = normalize_gender_prediction([[0.2, 0.8]])

    assert label == "Female"
    assert confidence == pytest.approx(0.8)


def test_gender_prediction_handles_binary_sigmoid_output():
    label, confidence = normalize_gender_prediction([[0.15]])

    assert label == "Male"
    assert confidence == pytest.approx(0.85)


def test_age_prediction_is_clipped_for_display():
    age, raw_age = normalize_age_prediction([[142.4]])

    assert age == 100
    assert raw_age == pytest.approx(142.4)
