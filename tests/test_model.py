"""Unit tests for the model code in starter/ml."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics, compute_slice_metrics, inference, train_model,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "census.csv"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def sample():
    """A small random sample of the cleaned census data."""
    data = pd.read_csv(DATA_PATH)
    return data.sample(n=2000, random_state=0)


@pytest.fixture(scope="module")
def processed(sample):
    """Processed features, labels, encoder and label binarizer."""
    return process_data(
        sample, categorical_features=CAT_FEATURES, label="salary",
        training=True,
    )


@pytest.fixture(scope="module")
def trained(processed):
    """A model trained on the processed sample."""
    X, y, _, _ = processed
    return train_model(X, y)


def test_process_data_returns_expected_types(sample, processed):
    X, y, encoder, lb = processed
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(sample) == y.shape[0]
    assert set(np.unique(y)) <= {0, 1}


def test_train_model_returns_random_forest(trained):
    assert isinstance(trained, RandomForestClassifier)


def test_inference_returns_binary_predictions(processed, trained):
    X, _, _, _ = processed
    preds = inference(trained, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_compute_slice_metrics_one_row_per_value(sample, processed, trained):
    _, _, encoder, lb = processed
    result = compute_slice_metrics(
        sample, "education", trained, encoder, lb, CAT_FEATURES)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == sample["education"].nunique()
    assert result["n"].sum() == len(sample)
    for col in ["precision", "recall", "f1"]:
        assert result[col].between(0, 1).all()
