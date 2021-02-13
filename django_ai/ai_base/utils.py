from django.contrib.contenttypes.models import ContentType


def get_model(app_model_str):
    app, model = app_model_str.split(".")
    model_class = ContentType.objects.get(
        app_label=app,
        model=model.lower()
    ).model_class()
    return(model_class)


def allNotNone(iterable):
    for x in iterable:
        if x is None:
            return False
    return True
