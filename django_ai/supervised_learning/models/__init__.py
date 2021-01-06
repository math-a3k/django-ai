# -*- coding: utf-8 -*-

import os

if 'DJANGO_TEST' in os.environ:
    from django_ai.supervised_learning.models.svm import SVC
    from django_ai.supervised_learning.models.hgb import HGBTree
else:
    from .svm import SVC
    from .hgb import HGBTree
