# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_to_file
from data_slice import data_slice

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
df = pd.read_csv('./data/census_no_spaces.csv')


def prepare_data(data):
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
    feature = test["marital-status"].to_numpy()

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, test_encoder, test_lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    save_to_file(encoder, "encoder")
    save_to_file(lb, "labelbinarizer")

    return X_train, y_train, X_test, y_test, feature


def train(X_train, y_train, X_test, y_test, feature):
    # Train and save a model.
    model = train_model(X_train, y_train)

    save_to_file(model, 'naive_bias.sav')

    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"precision: {str(precision)} recall {str(recall)} fbeta {str(fbeta)}")

    data_slice(feature, y_test, preds)


X_train, y_train, X_test, y_test, feature = prepare_data(df)

train(X_train, y_train, X_test, y_test, feature)
