# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Yuliia Kanarovska created the Naive Bias model using the default hyperparameters in scikit-learn 0.24.2
* Model version 1.0.0
* Model data 05/07/2021

## Intended Use
* This model should be used to predict the Census Income data (>50K or <=50K) based on the list of different descriptive attributes.
* Intended to be used as an academic project.

## Training Data
* The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
* The original dataset contains 48842 rows, and after an 80-20 split 80% of the data was used for training.
* For encoding features was used OneHotEncoder, for encoding labels was used LabelBinarizer.

## Evaluation Data
* 20% of the original data was used for testing purposes.

## Metrics
* Evaluation metrics include precision, recall, and fbeta. Where explains what portion of positives was correct; recall explains how many true positives were identified correctly and fbeata is is the weighted harmonic mean of precision and recall.
* Proposed model achieved precision: 0.67 recall 0.32 fbeta 0.43

## Ethical Considerations
* Demographic data were obtained from the public 1994 Census Database. No new information is inferred or annotated.

## Caveats and Recommendations
* Would be appropriate to do feature engineering and verify the distributions.
* Given gender classes are binary (male/female), would be better to obtain the range of genders for further evaluation.