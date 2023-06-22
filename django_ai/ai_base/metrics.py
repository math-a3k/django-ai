from .utils import load_settings_dicts

METRICS = load_settings_dicts("DJANGO_AI_METRICS")

METRICS_FORMATS = load_settings_dicts("DJANGO_AI_METRICS_FORMATS")

METRICS_FORMATS_MAPPING = load_settings_dicts(
    "DJANGO_AI_METRICS_FORMATS_MAPPING"
)


def format_metric(metric, values):
    formats_mapping = METRICS_FORMATS_MAPPING
    metric_format = formats_mapping.get(metric, METRICS_FORMATS["default"])
    if len(values) == 1:
        format_str = metric_format["single"]
    elif len(values) > 1:
        format_str = metric_format["many"]
    else:
        return None
    return format_str(values)
