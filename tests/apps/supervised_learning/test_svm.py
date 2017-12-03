#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_svm.py
------------

Tests for `django-ai` `svm` module.
"""
import random
import numpy as np

from django.test import TestCase
# from django.contrib.contenttypes.models import ContentType
# from django.core.exceptions import ValidationError

from django_ai.supervised_learning.models import svm

# from tests.test_models import models as test_models


class TestSVM(TestCase):

    def setUp(self):
        # Set the seeds
        random.seed(123456)
        np.random.seed(123456)
        # -> SVM 1
        self.svm1, _ = svm.SVC.objects.get_or_create(
            name="svm1"
        )

    def test_svm_engine_object(self):
        print("entro acá")
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        classifier = self.svm1.get_engine_object()
        # import ipdb; ipdb.set_trace()
        classifier.fit(X, y)
        self.assertEqual(classifier.predict([[-0.8, -1]]), [1])
