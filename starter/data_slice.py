import numpy as np
from ml.model import compute_model_metrics
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def data_slice(feature, y_test, preds):
    data = np.column_stack((feature, y_test, preds))

    for cls in np.unique(feature):
        fltr = np.asarray([cls])
        filtered = data[np.in1d(data[:, 0], fltr)]
        precision, recall, fbeta = compute_model_metrics(filtered[:, 1].astype(int), filtered[:, 2].astype(int))
        logger.info(f"For {cls} - precision: {str(precision)} recall {str(recall)} fbeta {str(fbeta)}")
        with open('slice_output.txt', 'a') as f:
            f.write(f"For {cls} - precision: {str(precision)} recall {str(recall)} fbeta {str(fbeta)}\n")
