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

from django.test import (TestCase, Client, )
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.template import Context, Template
from django.urls import reverse
from django.contrib.auth.models import User

from django_ai.base.models import (DataColumn, )
from django_ai.bayesian_networks import models as bn_models
from django_ai.bayesian_networks import bayespy_constants as bp_consts
from tests.test_models import models as test_models


class TestBase(TestCase):

    def setUp(self):
        # Set the seeds
        random.seed(123456)
        np.random.seed(123456)
        # Set up the user
        self.user, _ = User.objects.get_or_create(
            username='testadmin', email='testadmin@example.com',
            is_superuser=True
        )
        self.user.set_password("12345")
        self.user.save()
        self.client.login(username='testadmin', password='12345')

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
        # BN 1
        self.bn1, _ = bn_models.BayesianNetwork.objects.get_or_create(
            name="BN for tests - 1")
        self.mu, _ = bn_models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn1,
            name="mu",
            node_type=bn_models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=bp_consts.DIST_GAUSSIAN_ARD,
            distribution_params="0, 1e-6",
            graph_interval="-10, 20"
        )
        self.tau, _ = bn_models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn1,
            name="tau",
            node_type=bn_models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=bp_consts.DIST_GAMMA,
            distribution_params="1e-6, 1e-6",
            graph_interval="1e-6, 0.1"
        )
        self.ui_avg1, _ = bn_models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn1,
            name="userinfo.avg1",
            node_type=bn_models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=True,
            distribution=bp_consts.DIST_GAUSSIAN_ARD,
            distribution_params="mu, tau",
        )
        self.ui_avg1_col, _ = \
            bn_models.BayesianNetworkNodeColumn.objects.get_or_create(
                node=self.ui_avg1,
                ref_model=ContentType.objects.get(model="userinfo",
                                                  app_label="test_models"),
                ref_column="avg1",
            )
        self.e1, _ = bn_models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn1,
            description="mu -> userinfo.avg1",
            parent=self.mu,
            child=self.ui_avg1
        )
        self.e2, _ = bn_models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn1,
            description="tau -> userinfo.avg1",
            parent=self.tau,
            child=self.ui_avg1
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
        # Test action_url tag
        context = Context({
            'bn': self.bn1
        })
        template_to_render = Template(
            '{% load admin_extras %}'
            '{% action_url "perform_inference" bn %}'
            '{% action_url "reset_inference" bn %}'
            '{% action_url "reinitialize_rng" %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn(
            '/django-ai/run-action/perform_inference/bayesiannetwork/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reset_inference/bayesiannetwork/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reinitialize_rng',
            rendered_template
        )
        # Test ai_actions tag
        context = Context({
            'original': self.bn1
        })
        template_to_render = Template(
            '{% load admin_extras %}'
            '{% ai_actions %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn(
            '/django-ai/run-action/perform_inference/bayesiannetwork/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reset_inference/bayesiannetwork/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reinitialize_rng',
            rendered_template
        )

    def test_views(self):
        self.setUp()
        # -> Test perform_inference view
        # Test correct inference
        url = reverse('run-action', kwargs={
            "action": "perform_inference", "content_type": "bayesiannetwork",
            "object_id": self.bn1.id, }
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.bn1.refresh_from_db()
        self.assertTrue(self.bn1.engine_object is not None)
        # Test incorrect inference
        self.mu.distribution_params = "xxx:bad-dp"
        self.mu.save()
        referer = '/admin/bayesian_networks/bayesiannetwork/{}/change/'\
            .format(self.bn1.id)
        url = reverse('run-action', kwargs={
            "action": "perform_inference", "content_type": "bayesiannetwork",
            "object_id": self.bn1.id, }
        )
        # For some reason, the request must be done twice, otherwise it won't
        # update the response :\
        new_response = self.client.get(url, follow=True, HTTP_REFERER=referer)
        new_response = self.client.get(url, follow=True, HTTP_REFERER=referer)
        self.assertEqual(new_response.status_code, 200)
        message = list(new_response.context.get('messages'))[0]
        self.assertEqual(message.tags, "error")
        self.assertTrue("ERROR WHILE PERFORMING INFERENCE" in message.message)

        # -> Test reset_inference view
        url = reverse('run-action', kwargs={
            "action": "reset_inference", "content_type": "bayesiannetwork",
            "object_id": self.bn1.id, }
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.bn1.refresh_from_db()
        self.assertTrue(self.bn1.engine_object is None)

        # -> Test reinitialize_rng view
        state = random.getstate()
        url = reverse('run-action', kwargs={
            "action": "reinitialize_rng", }
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        new_state = random.getstate()
        self.assertTrue(state is not new_state)
