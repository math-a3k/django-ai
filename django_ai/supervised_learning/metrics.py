import numpy as np

METRICS_FORMATS = {
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

METRICS_FORMATS_MAPPING = {
    "accuracy": METRICS_FORMATS['percentage'],
    "precision": METRICS_FORMATS['percentage'],
    "recall": METRICS_FORMATS['percentage'],
}


def format_metric(metric, values):
    metric_format = \
        METRICS_FORMATS_MAPPING.get(metric, METRICS_FORMATS["default"])
    if len(values) == 1:
        format_str = metric_format["single"]
    elif len(values) > 1:
        format_str = metric_format["many"]
    else:
        return None
    return format_str(values)
