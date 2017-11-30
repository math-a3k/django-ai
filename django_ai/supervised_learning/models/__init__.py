# -*- coding: utf-8 -*-

import os

if 'DJANGO_TEST' in os.environ:
    from django_ai.supervised_learning.models.svm import SVC
else:
    from supervised_learning.models.svm import SVC
