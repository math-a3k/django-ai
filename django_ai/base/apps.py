# -*- coding: utf-8 -*-

import os

from django.apps import AppConfig


class BaseConfig(AppConfig):
    name = 'base'
    verbose_name = '[django-ai] Base'


if 'DJANGO_TEST' in os.environ:
    BaseConfig.name = 'django_ai.base'
