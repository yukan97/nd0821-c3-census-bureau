# Put the code for your API here.
from fastapi import Body, FastAPI
from pydantic import BaseModel
import pandas as pd
import os

from starter.ml.data import process_data
from starter.ml.model import inference, load_from_file


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# result = loaded_model.score(X_test, Y_test)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Instantiate the app.
app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"Greetings!"}


@app.post("/inference/")
async def create_item(data: Data = Body(None,
                                        example={
                                            "age": 39,
                                            "workclass": "State-gov",
                                            "fnlgt": 77516,
                                            "education": "Bachelors",
                                            "education_num": 13,
                                            "marital_status": "Never-married",
                                            "occupation": "Adm-clerical",
                                            "relationship": "Not-in-family",
                                            "race": "White",
                                            "sex": "Male",
                                            "capital_gain": 2174,
                                            "capital_loss": 0,
                                            "hours_per_week": 40,
                                            "native_country": "United-States"
                                        }
                                        )):
    dict = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education-num": [data.education_num],
        "marital-status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital-gain": [data.capital_gain],
        "capital-loss": [data.capital_loss],
        "hours-per-week": [data.hours_per_week],
        "native-country": [data.native_country]
    }

    data = pd.DataFrame.from_dict(dict)

    loaded_encoder = load_from_file('encoder')
    loaded_model = load_from_file('naive_bias.sav')
    loaded_lb = load_from_file('labelbinarizer')

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=loaded_encoder
    )
    preds = inference(loaded_model, X.reshape(1,108))
    preds = loaded_lb.inverse_transform(preds)
    return preds[0]
