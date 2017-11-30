# -*- coding: utf-8 -*-

import os

from django.apps import AppConfig


class SupervisedLearningConfig(AppConfig):
    name = 'supervised_learning'
    verbose_name = '[django_ai] Supervised Learning'


if 'DJANGO_TEST' in os.environ:
    SupervisedLearningConfig.name = 'django_ai.supervised_learning'
