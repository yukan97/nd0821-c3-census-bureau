# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from data_slice import data_slice
import pickle
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
data = pd.read_csv('./data/census_no_spaces.csv')
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

with open("encoder", "wb") as f:
    pickle.dump(encoder, f)

# Train and save a model.
model = train_model(X_train, y_train)
logger.info(type(model))

preds = inference(model, X_test)

print(y_test.shape)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"precision: {str(precision)} recall {str(recall)} fbeta {str(fbeta)}")

data_slice(feature, y_test, preds)
# save the model to disk
with open('naive_bias.pkl', 'wb') as f:
    pickle.dump(model, f)

