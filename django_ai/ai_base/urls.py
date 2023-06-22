from django.urls import path

from . import views


urlpatterns = [
    path(
        "run-action/<action>/<content_type>/<int:object_id>",
        views.RunActionView.as_view(),
        name="run-action",
    ),
    path(
        "run-action/<action>", views.RunActionView.as_view(), name="run-action"
    ),
]
