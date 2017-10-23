# -*- coding: utf-8 -*-

from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^run_inference/(?P<bn_id>[0-9]+)$',
        views.bn_run_inference,
        name="bn_run_inference"),
    url(r'^reset_inference/(?P<bn_id>[0-9]+)$',
        views.bn_reset_inference,
        name="bn_reset_inference"),
    url(r'^reinitialize_rng/$',
        views.bn_reinitialize_rng,
        name="bn_reinitialize_rng"),
]
