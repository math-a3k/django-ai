from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.utils.module_loading import import_string


def get_model(app_model_str):
    app, model = app_model_str.split(".")
    model_class = ContentType.objects.get(
        app_label=app, model=model.lower()
    ).model_class()
    return model_class


def allNotNone(iterable):
    for x in iterable:
        if x is None:
            return False
    return True


def load_settings_dicts(setting_name):
    collected = {}
    for dotted_path in getattr(settings, setting_name):
        imported_dict = import_string(dotted_path)
        collected = {**collected, **imported_dict}
    return collected
