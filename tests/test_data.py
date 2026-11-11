
from src.data import load_dataset


def test_dataset_has_expected_shape():

    X, y = load_dataset()

    assert X.shape == (569, 30)
    assert y.shape == (569,)


def test_dataset_contains_two_classes():

    X, y = load_dataset()

    assert set(y) == {0, 1}
