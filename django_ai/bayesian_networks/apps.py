# -*- coding: utf-8 -*-

import os

from django.apps import AppConfig


class BayesianNetworksConfig(AppConfig):
    name = 'bayesian_networks'
    verbose_name = '[django-ai] Bayesian Networks'


if 'DJANGO_TEST' in os.environ:
    BayesianNetworksConfig.name = 'django_ai.bayesian_networks'
