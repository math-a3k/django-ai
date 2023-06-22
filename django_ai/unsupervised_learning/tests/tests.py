#
"""
Tests for `django-ai` `ai-base` module.
====================================
"""
import random
import numpy as np
from copy import deepcopy
from unittest import mock

# from django.apps.registry import apps as dapps
from django.core.management import call_command

# from django.contrib.contenttypes.management import create_contenttypes
# from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.test import Client, SimpleTestCase, modify_settings

# from django.template import Context, Template
from django.contrib.auth.models import User

# from django.urls import reverse
# from django.utils import timezone

# from django_ai.ai_base.models import DataColumn
from django_ai.unsupervised_learning.models import (
    UnsupervisedLearningTechnique,
)

# from django_ai.ai_base import utils as ai_utils

from .ult_test_models import models as ult_test_models


@modify_settings(
    INSTALLED_APPS={
        "append": "django_ai.unsupervised_learning.tests.ult_test_models",
    }
)
class TestUnsupervisedLearning(SimpleTestCase):
    """
    SimpleTestCase is needed otherwise migrations won't run inside transactions in SQLite
    in Django 3.1
    """

    databases = "__all__"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Hackery for ensuring all the cts are available
        # apps = dapps
        # tm_config = apps.get_app_config('ult_test_models')
        # tm_config.models_module = tm_config.models_module or True
        # create_contenttypes(tm_config, verbosity=0)
        # Run migrations
        call_command("migrate", "ult_test_models", verbosity=0, no_color=True)

        # Set the seeds
        random.seed(123456)
        np.random.seed(123456)

        # Set up the user
        cls.user, _ = User.objects.get_or_create(
            username="testadmin",
            email="testadmin@example.com",
            is_superuser=True,
        )
        cls.user.set_password("12345")
        cls.user.save()
        cls.client = Client()
        cls.client.login(username="testadmin", password="12345")

        cls.ult, _ = UnsupervisedLearningTechnique.objects.get_or_create(
            name="Test Unsupervised Learning Technique",
            results_storage="dmf:ult_test_models.userinfo2.cluster_2",
            threshold_actions=":recalculate",
            data_model="ult_test_models.UserInfo",
            learning_fields="sex, avg1",
        )

    def test_ult_store_results(self):
        # -> Test with no results_storage
        self.ult.results_storage = None
        self.assertFalse(self.ult.store_results())
        results = np.random.choice(["A", "B"], 100)
        ult = deepcopy(self.ult)
        # -> Test mocking results
        with mock.patch.object(ult, "get_results", return_value=results):
            # -> Test with other results_storage engine
            ult.results_storage = (
                "otherengine:ult_test_models.userinfo2.cluster_2"
            )
            with self.assertRaises(ValueError):
                ult.store_results()
            # -> Test storing results
            self.setUp()
            ult.results_storage = "dmf:ult_test_models.userinfo2.cluster_2"
            ult.store_results()
            stored_results = ult_test_models.UserInfo2.objects.values_list(
                "cluster_2", flat=True
            )
            self.assertTrue(all(stored_results == results))
            # -> Test reset of results
            ult.store_results(reset=True)
            stored_results = ult_test_models.UserInfo2.objects.values_list(
                "cluster_2", flat=True
            )
            self.assertTrue(not any(stored_results))

    def test_ult_validation(self):
        # Test invalid syntax
        with self.assertRaises(ValidationError):
            self.ult.results_storage = "drf-examples.models.blabla"
            self.ult.full_clean()
        # Test invalid engine
        with self.assertRaises(ValidationError):
            self.ult.results_storage = "drf:examples.models.blabla"
            self.ult.full_clean()
        # Test 'dmf' invalid path
        with self.assertRaises(ValidationError):
            self.ult.results_storage = "dmf:examples.models"
            self.ult.full_clean()
        # Test 'dmf' invalid model
        with self.assertRaises(ValidationError):
            self.ult.results_storage = (
                "dmf:ult_test_models.non-existant-model.n-e-field"
            )
            self.ult.full_clean()
        # Test 'dmf' invalid field
        with self.assertRaises(ValidationError):
            self.ult.results_storage = "dmf:ult_test_models.UserInfo.n-e-field"
            self.ult.full_clean()
        # Test 'dmf' correct content
        self.ult.results_storage = "dmf:ult_test_models.UserInfo.cluster_1"
        self.assertEqual(self.ult.full_clean(), None)
        # Test no results_storage
        self.ult.results_storage = None
        self.assertEqual(self.ult.full_clean(), None)
