"""FastAPI app that serves the census income model."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact
from starter.train_model import CAT_FEATURES

MODEL_DIR = Path(__file__).resolve().parent / "model"

model = load_artifact(MODEL_DIR / "model.joblib")
encoder = load_artifact(MODEL_DIR / "encoder.joblib")
lb = load_artifact(MODEL_DIR / "lb.joblib")

app = FastAPI(
    title="Census Income Predictor",
    description="Predicts whether a person earns more than $50K a year.",
    version="1.0.0",
)


class CensusRecord(BaseModel):
    """One person's census attributes (the CSV columns, minus salary)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 38,
                    "workclass": "Private",
                    "fnlgt": 215646,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Divorced",
                    "occupation": "Handlers-cleaners",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class Prediction(BaseModel):
    """The model's answer for one record."""

    prediction: str = Field(examples=["<=50K"])


@app.get("/")
async def welcome() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(record: CensusRecord) -> Prediction:
    """Predict whether the person earns <=50K or >50K."""
    df = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, training=False,
        encoder=encoder, lb=lb,
    )
    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]
    return Prediction(prediction=str(label))
