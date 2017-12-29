# -*- coding: utf-8 -*-
import os

from django.apps import AppConfig


class SpamFilteringConfig(AppConfig):
    name = 'systems.spam_filtering'
    verbose_name = '[django-ai] Systems'


if 'DJANGO_TEST' in os.environ:
    SpamFilteringConfig.name = 'django_ai.systems.spam_filtering'
