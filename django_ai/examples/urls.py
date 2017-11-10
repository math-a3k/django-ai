# -*- coding: utf-8 -*-

from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^pages$',
        views.a_page_of_type_X,
        name="page"),
    url(r'^pages/(?P<page_type>[A-J])/$',
        views.a_page_of_type_X,
        name="page"),
    url(r'^pages/(?P<page_type>[A-J])/(?P<user_id>[0-9]+)$',
        views.a_page_of_type_X,
        name="page"),
    url(r'^new-user$',
        views.new_user,
        name="new-user"),
    url(r'^metrics$',
        views.process_metrics,
        name="process-metrics"),
]
