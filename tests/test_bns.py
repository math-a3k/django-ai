#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_django-ai
------------

Tests for `django-ai` models module.
"""
import numpy as np

from django.test import TestCase
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError

from django_ai.bayesian_networks import models
from django_ai.bayesian_networks.bayespy_constants import (
    DIST_GAUSSIAN_ARD, DIST_GAMMA, DIST_GAUSSIAN, DET_ADD)
from tests.test_models.models import UserInfo

class TestDjango_ai(TestCase):

    def setUp(self):
        # BN 1
        self.bn1 = models.BayesianNetwork.objects.create(name="BN for tests - 1")
        self.mu = models.BayesianNetworkNode.objects.create(
            network=self.bn1,
            name="mu",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN_ARD,
            distribution_params="0, 1e-6",
            graph_interval="-10, 20"
        )
        self.tau = models.BayesianNetworkNode.objects.create(
            network=self.bn1,
            name="tau",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAMMA,
            distribution_params="1e-6, 1e-6",
            graph_interval="1e-6, 0.1"
        )
        self.ui_avg1 = models.BayesianNetworkNode.objects.create(
            network=self.bn1,
            name="userinfo.avg1",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=True,
            distribution=DIST_GAUSSIAN_ARD,
            distribution_params="mu, tau",
            ref_model=ContentType.objects.get(model="userinfo",
                                              app_label="test_models"),
            ref_column="avg1",
        )
        self.e1 = models.BayesianNetworkEdge.objects.create(
            network=self.bn1,
            description="mu -> userinfo.avg1",
            parent=self.mu,
            child=self.ui_avg1
        )
        self.e2 = models.BayesianNetworkEdge.objects.create(
            network=self.bn1,
            description="tau -> userinfo.avg1",
            parent=self.tau,
            child=self.ui_avg1
        )
        # BN 2
        self.bn2 = models.BayesianNetwork.objects.create(name="BN for tests - 2")
        self.x1 = models.BayesianNetworkNode.objects.create(
            network=self.bn2,
            name="x1",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN,
            distribution_params="[0, 0], [[1, 0], [0,1]]",
        )
        self.x2 = models.BayesianNetworkNode.objects.create(
            network=self.bn2,
            name="x2",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN,
            distribution_params="[1, 1], [[1, 0], [0,1]]",
        )
        self.z = models.BayesianNetworkNode.objects.create(
            network=self.bn2,
            name="z",
            node_type=models.BayesianNetworkNode.NODE_TYPE_DETERMINISTIC,
            is_observable=False,
            deterministic=DET_ADD,
            deterministic_params="x1, x2",
        )
        self.bn2e1 = models.BayesianNetworkEdge.objects.create(
            network=self.bn2,
            description="x1 -> z",
            parent=self.x1,
            child=self.z
        )
        self.bn2e2 = models.BayesianNetworkEdge.objects.create(
            network=self.bn2,
            description="x2 -> z",
            parent=self.x2,
            child=self.z
        )

    def test_bn_inference(self):
        self.bn1.perform_inference(recalculate=True)
        Q = self.bn1.engine_object
        mu = Q['mu'].get_moments()[0]
        tau = Q['tau'].get_moments()[0]
        # For avoiding rounding and float differences
        self.assertEqual(str(mu)[:5], '9.809')
        self.assertEqual(str(tau)[:5], '0.039')

    def test_bn_node_validation(self):
        ## Test First Step: fields corresponds to Node type
        with self.assertRaises(ValidationError):
            self.mu.deterministic_params = "a, b"
            self.mu.full_clean()
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.node_type = \
                models.BayesianNetworkNode.NODE_TYPE_DETERMINISTIC
            self.mu.full_clean()

        ## Test Second Step: Validations on Stochastic Types
        # Observables must be linked to a model
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1.ref_model = None
            self.ui_avg1.full_clean()
        # Observables must be linked to a field of a model
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1.ref_column = None
            self.ui_avg1.full_clean()
        # Observables must be linked to an existing field of a model
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1.ref_column = "non-existant-field"
            self.ui_avg1.full_clean()
        # If not Observable, ref_model and ref_column musts be empty
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1.is_observable = False
            self.ui_avg1.full_clean()
        # Stochastic Nodes must have a Distribution
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.distribution = None
            self.mu.full_clean()
        # Stochastic Nodes must have a Distribution Params
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.distribution_params = None
            self.mu.full_clean()

        ## Test Third Step: Validations on Deterministic Types
        # Deterministic Nodes must have a function
        self.setUp()
        with self.assertRaises(ValidationError):
            self.z.deterministic = None
            self.z.full_clean()
        # Deterministic Nodes must have function parameters
        self.setUp()
        with self.assertRaises(ValidationError):
            self.z.deterministic_params = None
            self.z.full_clean()

        ## Test Final Step: BayesPy initialization
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.distribution_params = "1, 2, 3, 4, 5"
            self.mu.full_clean()


    def tearDown(self):
        self.bn1.image.delete()
        self.mu.image.delete()
        self.tau.image.delete()
