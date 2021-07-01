# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle

# Add code to load in the data.
data = pd.read_csv('./data/census_no_space.csv')

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, test_encoder, test_lb = process_data(
    test, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

# save the model to disk
pickle.dump(model, open('naive_bias.pkl', 'wb'))

