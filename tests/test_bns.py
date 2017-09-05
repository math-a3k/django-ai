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

from django_ai.bayesian_networks import models
from django_ai.bayesian_networks.bayespy_constants import (
    DIST_GAUSSIAN_ARD, DIST_GAMMA)
from tests.test_models.models import UserInfo

class TestDjango_ai(TestCase):

    def setUp(self):
        self.bn = models.BayesianNetwork.objects.create(name="BN for tests")
        self.mu = models.BayesianNetworkNode.objects.create(
            network=self.bn,
            name="mu",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN_ARD,
            distribution_params="0, 1e-6",
            graph_interval="-10, 20"
        )
        self.tau = models.BayesianNetworkNode.objects.create(
            network=self.bn,
            name="tau",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAMMA,
            distribution_params="1e-6, 1e-6",
            graph_interval="1e-6, 0.1"
        )
        self.ui_avg1 = models.BayesianNetworkNode.objects.create(
            network=self.bn,
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
            network=self.bn,
            description="mu -> userinfo.avg1",
            parent=self.mu,
            child=self.ui_avg1
        )
        self.e2 = models.BayesianNetworkEdge.objects.create(
            network=self.bn,
            description="tau -> userinfo.avg1",
            parent=self.tau,
            child=self.ui_avg1
        )

    def test_bn_inference(self):
        self.bn.perform_inference()
        Q = self.bn.engine_object
        mu = Q['mu'].get_moments()[0]
        tau = Q['tau'].get_moments()[0]
        # For avoiding rounding and float differences
        self.assertEqual(str(mu)[:5], '9.809')
        self.assertEqual(str(tau)[:5], '0.039')

    def tearDown(self):
        self.bn.image.delete()
        self.mu.image.delete()
        self.tau.image.delete()
