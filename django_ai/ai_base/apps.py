# -*- coding: utf-8 -*-

import os

from django.apps import AppConfig


class AIBaseConfig(AppConfig):
    name = 'django_ai.ai_base'
    verbose_name = '[django-ai] Base'


if 'DJANGO_TEST' in os.environ:
    AIBaseConfig.name = 'django_ai.ai_base'
