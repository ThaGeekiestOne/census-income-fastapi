"""Send one POST request to the deployed API and print the result."""
import sys

import requests

LIVE_URL = "https://census-income-fastapi-8a7l.onrender.com"

PERSON = {
    "age": 52,
    "workclass": "Private",
    "fnlgt": 99185,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Sales",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 7298,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}


def main(base_url=LIVE_URL):
    """POST one record to /predict and print the response."""
    response = requests.post(f"{base_url}/predict", json=PERSON, timeout=120)
    print("URL:", f"{base_url}/predict")
    print("HTTP status code:", response.status_code)
    print("Inference result:", response.json())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else LIVE_URL)
