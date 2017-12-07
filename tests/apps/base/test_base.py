#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_base.py
------------

Tests for `django-ai` `base` module.
"""
import random
import numpy as np
from unittest import mock

from django.test import TestCase
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.template import Context, Template

from django_ai.base.models import (DataColumn, )
from tests.test_models import models as test_models


class TestBase(TestCase):

    def setUp(self):
        # Set the seeds
        random.seed(123456)
        np.random.seed(123456)

        self.mystatmodel, _ = test_models.MyStatisticalModel.objects\
            .get_or_create(
                name="My Stat Model",
                has_results=True,
                results_storage="dmf:test_models.userinfo2.cluster_2",
                threshold_actions=":recalculate"
            )
        self.dc_ui_avg1 = DataColumn(
            content_type=ContentType.objects.get_for_model(
                test_models.MyStatisticalModel),
            object_id=self.mystatmodel.id,
            ref_model=ContentType.objects.get(model="userinfo",
                                              app_label="test_models"),
            ref_column="avg1",
            position=0
        )
        self.dc_ui2_avg2 = DataColumn(
            content_type=ContentType.objects.get_for_model(
                test_models.MyStatisticalModel),
            object_id=self.mystatmodel.id,
            ref_model=ContentType.objects.get(model="userinfo2",
                                              app_label="test_models"),
            ref_column="avg2",
            position=1
        )
        self.mysltechnique, _ = test_models.MySupervisedLearningTechnique\
            .objects.get_or_create(
                name="My Supervised Learning Technique",
            )
        self.myultechnique, _ = test_models.MyUnsupervisedLearningTechnique\
            .objects.get_or_create(
                name="My Unsupervised Learning Technique",
            )

    def test_statistical_model_get_data(self):
        # -> Test with no data columns
        with self.assertRaises(ValueError):
            self.mystatmodel.get_data()
        # -> Add the column for testing with one column
        self.dc_ui_avg1.save()
        expected_output = [
            [avg] for avg in
            test_models.UserInfo.objects.values_list("avg1", flat=True)
        ]
        self.assertTrue(all(self.mystatmodel.get_data() == expected_output))
        # -> Add another column with different size
        self.dc_ui2_avg2.save()
        with self.assertRaises(ValidationError):
            self.mystatmodel.get_data()

    def test_statistical_model_store_results(self):
        # -> Test with no results_storage
        self.mystatmodel.results_storage = None
        self.assertFalse(self.mystatmodel.store_results())
        results = np.random.choice(["A", "B"], 100)
        my_sm = self.mystatmodel
        # -> Test mocking results
        with mock.patch.object(my_sm, 'get_results', return_value=results):
            # -> Test with other results_storage engine
            self.mystatmodel.results_storage = \
                "otherengine:test_models.userinfo2.cluster_2"
            with self.assertRaises(ValueError):
                self.mystatmodel.store_results()
            # -> Test storing results
            self.setUp()
            my_sm.results_storage = "dmf:test_models.userinfo2.cluster_2"
            my_sm.store_results()
            stored_results = test_models.UserInfo2.objects.values_list(
                "cluster_2", flat=True)
            self.assertTrue(all(stored_results == results))
            # -> Test reset of results
            my_sm.store_results(reset=True)
            stored_results = test_models.UserInfo2.objects.values_list(
                "cluster_2", flat=True)
            self.assertTrue(not any(stored_results))

    def test_statistical_model_validation(self):
        # Test invalid syntax
        with self.assertRaises(ValidationError):
            self.mystatmodel.results_storage = "drf-examples.models.blabla"
            self.mystatmodel.full_clean()
        # Test invalid engine
        with self.assertRaises(ValidationError):
            self.mystatmodel.results_storage = "drf:examples.models.blabla"
            self.mystatmodel.full_clean()
        # Test 'dmf' invalid path
        with self.assertRaises(ValidationError):
            self.mystatmodel.results_storage = "dmf:examples.models"
            self.mystatmodel.full_clean()
        # Test 'dmf' invalid model
        with self.assertRaises(ValidationError):
            self.mystatmodel.results_storage = \
                "dmf:test_models.non-existant-model.n-e-field"
            self.mystatmodel.full_clean()
        # Test 'dmf' invalid field
        with self.assertRaises(ValidationError):
            self.mystatmodel.results_storage = \
                "dmf:test_models.UserInfo.n-e-field"
            self.mystatmodel.full_clean()
        # Test 'dmf' correct content
        self.mystatmodel.results_storage = "dmf:test_models.UserInfo.cluster_1"
        self.assertEqual(self.mystatmodel.full_clean(), None)
        # Test threshold actions incorrect content
        with self.assertRaises(ValidationError):
            self.mystatmodel.threshold_actions = \
                ":recalculate :not-supported-kw"
            self.mystatmodel.full_clean()

    def test_slt_get_labels(self):
        # Test no labels column defined
        self.assertTrue(self.mysltechnique.get_labels() is None)
        # Test labels column correct content
        self.mysltechnique.labels_column = "test_models.UserInfo2.cluster_2"
        expected_labels = test_models.UserInfo2.objects.values_list(
            "cluster_2", flat=True)
        output_labels = self.mysltechnique.get_labels()
        self.assertTrue(list(expected_labels) == list(output_labels))

    def test_slt_validation(self):
        # Test invalid syntax
        with self.assertRaises(ValidationError):
            self.mysltechnique.labels_column = \
                "test_models:userinfo2:cluster_2"
            self.mysltechnique.full_clean()
        # Test invalid model
        with self.assertRaises(ValidationError):
            self.mysltechnique.labels_column = \
                "test_models.non-existant-model.n-e-field"
            self.mysltechnique.full_clean()
        # Test invalid attribute
        with self.assertRaises(ValidationError):
            self.mysltechnique.labels_column = \
                "test_models.userinfo2.n-e-field"
            self.mysltechnique.full_clean()
        # Test 'dmf' correct content
        self.mysltechnique.labels_column = \
            "test_models.UserInfo2.cluster_2"
        self.assertEqual(self.mysltechnique.full_clean(), None)

    def test_datacolumn_validation(self):
        # Test invalid model
        with self.assertRaises(ValidationError):
            self.dc_ui_avg1.ref_model_id = 100
            self.dc_ui_avg1.ref_model = None
            self.dc_ui_avg1.full_clean()
        # Test invalid column
        with self.assertRaises(ValidationError):
            self.dc_ui2_avg2.ref_column = "avg3"
            self.dc_ui2_avg2.full_clean()
        # Test correct content
        self.dc_ui2_avg2.ref_column = "avg2"
        self.assertEqual(self.dc_ui2_avg2.full_clean(), None)

    def test_templatetags(self):
        # Test getitem filter
        context = Context({
            'mydict': {'key1': 'value1', 'key2': 'value2'}
        })
        template_to_render = Template(
            '{% load admin_extras %}'
            '{{ mydict|get_item:"key2" }}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('value2', rendered_template)
