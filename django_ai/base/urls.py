# -*- coding: utf-8 -*-

import os

from django.conf.urls import (url, include)

from . import views

examples_urls = 'examples.urls'
if 'DJANGO_TEST' in os.environ:
    examples_urls = 'django_ai.examples.urls'

urlpatterns = [
    url((r'^run-action/(?P<action>[\w_]+)/'
         r'(?P<content_type>[\w_]+)/'
         r'(?P<object_id>[0-9]+)$'),
        views.RunActionView.as_view(),
        name="run-action"),
    url((r'^run-action/(?P<action>[\w_]+)$'),
        views.RunActionView.as_view(),
        name="run-action"),
    # Examples
    url(r'^examples/', include(examples_urls)),
]
