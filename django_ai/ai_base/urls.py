from django.conf.urls import url

from . import views


urlpatterns = [
    url((r'^run-action/(?P<action>[\w_]+)/'
         r'(?P<content_type>[\w_]+)/'
         r'(?P<object_id>[0-9]+)$'),
        views.RunActionView.as_view(),
        name="run-action"),
    url((r'^run-action/(?P<action>[\w_]+)$'),
        views.RunActionView.as_view(),
        name="run-action"),
]
