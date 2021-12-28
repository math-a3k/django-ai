import numpy as np


BASE_METRICS_FORMATS = {
    "default": {
        "single": lambda values: values[0],
        "many": lambda values: (
            "{:03.3f} +/- {:03.3f}".format(np.mean(values), 2 * np.std(values))
        )
    },
    "percentage": {
        "single": lambda values: "{:03.1f}%".format(values[0]),
        "many": lambda values: (
            "{:03.1f}% +/- {:03.1f}%"
            .format(np.mean(values) * 100, 200 * np.std(values))
        )
    }
}

BASE_METRICS_FORMATS_MAPPING = {
    "accuracy": BASE_METRICS_FORMATS['percentage'],
    "precision": BASE_METRICS_FORMATS['percentage'],
    "recall": BASE_METRICS_FORMATS['percentage'],
}
