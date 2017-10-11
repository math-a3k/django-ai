# -*- coding: utf-8
from __future__ import unicode_literals, absolute_import

import os
import django
import django_ai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUG = True
USE_TZ = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "**************************************************"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

ROOT_URLCONF = "tests.urls"

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    'django_dag', # Dependency
    'nested_admin', # Dependency
    'django_ai.bayesian_networks.apps.BayesianNetworksConfig',
    # 'django_ai.examples',
    'tests.test_models'
]

SITE_ID = 1

if django.VERSION >= (1, 10):
    MIDDLEWARE = ()
else:
    MIDDLEWARE_CLASSES = ()


STATIC_URL = '/static/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'tests/media')
MEDIA_URL = '/media/'
