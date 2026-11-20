from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifact(obj, path):
    """Save a model, encoder or label binarizer to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_artifact(path):
    """Load a saved model, encoder or label binarizer."""
    return joblib.load(path)


def compute_slice_metrics(df, feature, model, encoder, lb,
                          cat_features, label="salary"):
    """Compute metrics for each unique value of `feature`, held fixed.

    Returns
    -------
    pd.DataFrame
        One row per unique value with count, precision, recall and F1.
    """
    rows = []
    for value in sorted(df[feature].unique()):
        part = df[df[feature] == value]
        X, y, _, _ = process_data(
            part, categorical_features=cat_features, label=label,
            training=False, encoder=encoder, lb=lb,
        )
        precision, recall, fbeta = compute_model_metrics(
            y, inference(model, X))
        rows.append({
            "feature": feature, "value": value, "n": len(part),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(fbeta, 4),
        })
    return pd.DataFrame(rows)
