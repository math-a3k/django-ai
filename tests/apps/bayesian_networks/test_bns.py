#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_django-ai
------------

Tests for `django-ai` models module.
"""
import random

import numpy as np
from bayespy.nodes import Gaussian

from django.test import (TestCase, Client, )
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.urls import reverse
from django.contrib.auth.models import User

from django_ai.bayesian_networks import models
from django_ai.bayesian_networks.bayespy_constants import (
    DIST_GAUSSIAN_ARD, DIST_GAMMA, DIST_GAUSSIAN, DET_ADD,
    DIST_DIRICHLET, DIST_WISHART, DIST_CATEGORICAL, DIST_MIXTURE, )
from django_ai.bayesian_networks.utils import (
    parse_node_args, mahalanobis_distance, )
from django_ai.bayesian_networks import views

from tests.test_models import models as test_models


class TestBN(TestCase):

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
        # BN 1
        self.bn1, _ = models.BayesianNetwork.objects.get_or_create(
            name="BN for tests - 1")
        self.mu, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn1,
            name="mu",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN_ARD,
            distribution_params="0, 1e-6",
            graph_interval="-10, 20"
        )
        self.tau, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn1,
            name="tau",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAMMA,
            distribution_params="1e-6, 1e-6",
            graph_interval="1e-6, 0.1"
        )
        self.ui_avg1, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn1,
            name="userinfo.avg1",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=True,
            distribution=DIST_GAUSSIAN_ARD,
            distribution_params="mu, tau",
        )
        self.ui_avg1_col, _ = \
            models.BayesianNetworkNodeColumn.objects.get_or_create(
                node=self.ui_avg1,
                ref_model=ContentType.objects.get(model="userinfo",
                                                  app_label="test_models"),
                ref_column="avg1",
            )
        self.e1, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn1,
            description="mu -> userinfo.avg1",
            parent=self.mu,
            child=self.ui_avg1
        )
        self.e2, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn1,
            description="tau -> userinfo.avg1",
            parent=self.tau,
            child=self.ui_avg1
        )
        # BN 2
        self.bn2, _ = models.BayesianNetwork.objects.get_or_create(
            name="BN for tests - 2")
        self.x1, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn2,
            name="x1",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN,
            distribution_params="[0, 0], [[1, 0], [0,1]]",
        )
        self.x2, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn2,
            name="x2",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN,
            distribution_params="[1, 1], [[1, 0], [0,1]]",
        )
        self.z, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn2,
            name="z",
            node_type=models.BayesianNetworkNode.NODE_TYPE_DETERMINISTIC,
            is_observable=False,
            deterministic=DET_ADD,
            deterministic_params="x1, x2",
        )
        self.bn2e1, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn2,
            description="x1 -> z",
            parent=self.x1,
            child=self.z
        )
        self.bn2e2, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn2,
            description="x2 -> z",
            parent=self.x2,
            child=self.z
        )
        # BN 3 (Clustering)
        self.bn3, _ = models.BayesianNetwork.objects.get_or_create(
            name="Clustering (testing)",
            network_type=models.BayesianNetwork.BN_TYPE_CLUSTERING,
            engine_meta_iterations=10,
            results_storage="dmf:test_models.userinfo.cluster_1",
            counter_threshold=2,
            threshold_actions=":recalculate",
        )
        self.alpha, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn3,
            name="alpha",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_DIRICHLET,
            distribution_params="numpy.full(10, 1e-05)",
        )
        self.Z, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn3,
            name="Z",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_CATEGORICAL,
            distribution_params="alpha, plates=(:dl_Y, ), :ifr",
        )
        self.mu_c, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn3,
            name="mu",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_GAUSSIAN,
            distribution_params=("numpy.zeros(2), [[1e-5,0], [0, 1e-5]], "
                                 "plates=(10, )"),
        )
        self.Lambda, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn3,
            name="Lambda",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=False,
            distribution=DIST_WISHART,
            distribution_params="2, [[1e-5,0], [0, 1e-5]], plates=(10, )",
        )
        self.Y, _ = models.BayesianNetworkNode.objects.get_or_create(
            network=self.bn3,
            name="Y",
            node_type=models.BayesianNetworkNode.NODE_TYPE_STOCHASTIC,
            is_observable=True,
            distribution=DIST_MIXTURE,
            distribution_params=("Z, @bayespy.nodes.Gaussian(), "
                                 "mu, Lambda, :noplates"),
        )
        #
        self.Y_col_avg_logged, _ = \
            models.BayesianNetworkNodeColumn.objects.get_or_create(
                node=self.Y,
                ref_model=ContentType.objects.get(
                    model="userinfo", app_label="test_models"),
                ref_column="avg_time_pages"
            )
        self.Y_col_avg_pages_a, _ = \
            models.BayesianNetworkNodeColumn.objects.get_or_create(
                node=self.Y,
                ref_model=ContentType.objects.get(
                    model="userinfo", app_label="test_models"),
                ref_column="avg_time_pages_a"
            )
        #
        self.alpha_to_Z, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn3,
            description="alpha -> Z",
            parent=self.alpha,
            child=self.Z
        )
        self.Z_to_Y, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn3,
            description="Z -> Y",
            parent=self.Z,
            child=self.Y
        )
        self.mu_to_Y, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn3,
            description="mu -> Y",
            parent=self.mu_c,
            child=self.Y
        )
        self.Lambda_to_Y, _ = models.BayesianNetworkEdge.objects.get_or_create(
            network=self.bn3,
            description="Lambda -> Y",
            parent=self.Lambda,
            child=self.Y
        )

    def test_bn_inference(self):
        self.bn1.perform_inference(recalculate=True)
        Q = self.bn1.engine_object
        mu = Q['mu'].get_moments()[0]
        tau = Q['tau'].get_moments()[0]
        # For avoiding rounding and float differences
        self.assertEqual(str(mu)[:5], '9.809')
        self.assertEqual(str(tau)[:5], '0.039')

    def test_bn_cached_eo(self):
        self.bn1.get_engine_object()
        expected_output = self.bn1.engine_object
        actual_output = self.bn1.get_engine_object()
        self.assertEqual(expected_output, actual_output)

    def test_ww_bn_reset_inference(self):
        """
        Django parallel test running has issues, the 'ww' in the test name
        is to make it run it at the end where no problems arise
        """
        self.setUp()
        expected_clean_metadata = {
            "clusters_labels": {},
            "prev_clusters_labels": {},
            "clusters_means": {},
            "prev_clusters_means": {},
            "clusters_sizes": {},
            "prev_clusters_sizes": {},
            "columns": [],
        }
        # Avoid unneccesary calculation
        self.bn3.engine_meta_iterations = 1
        #
        self.bn3.perform_inference()
        self.assertTrue(self.bn3.engine_object is not None)
        self.assertTrue(self.bn3.engine_object_timestamp is not None)
        self.assertTrue(self.bn3.metadata != expected_clean_metadata)
        results = test_models.UserInfo.objects.values_list(
            "cluster_1", flat=True)
        self.assertTrue(any(list(results)))
        self.bn3.reset_inference()
        self.assertTrue(self.bn3.engine_object is None)
        self.assertTrue(self.bn3.engine_object_timestamp is None)
        self.assertTrue(self.bn3.metadata == expected_clean_metadata)
        results = test_models.UserInfo.objects.values_list(
            "cluster_1", flat=True)
        self.assertTrue(not any(list(results)))

    def test_bn_deterministic_nodes(self):
        # Initialize the EO
        self.bn2.get_engine_object(reconstruct=True, save=True)
        self.z.refresh_from_db()
        z_eo = self.z.get_engine_object()
        expected_moments = [np.array([1., 1.]),
                            np.array([[3., 1.], [1., 3.]])]
        moments = z_eo.get_moments()
        self.assertTrue(all(expected_moments[0] == moments[0]))
        self.assertTrue(all(expected_moments[1][0] == moments[1][0]))
        self.assertTrue(all(expected_moments[1][1] == moments[1][1]))

    def test_bn_validation(self):
        # Test invalid syntax
        with self.assertRaises(ValidationError):
            self.bn3.results_storage = "drf-examples.models.blabla"
            self.bn3.full_clean()
        # Test invalid engine
        with self.assertRaises(ValidationError):
            self.bn3.results_storage = "drf:examples.models.blabla"
            self.bn3.full_clean()
        # Test 'dfm' invalid path
        with self.assertRaises(ValidationError):
            self.bn3.results_storage = "drf:examples.models"
            self.bn3.full_clean()
        # Test 'dfm' invalid model
        with self.assertRaises(ValidationError):
            self.bn3.results_storage = "drf:tests.non-existant-model"
            self.bn3.full_clean()
        # Test 'dfm' invalid field
        with self.assertRaises(ValidationError):
            self.bn3.results_storage = "drf:tests.UserInfo.n-e-field"
            self.bn3.full_clean()
        # Test 'dfm' correct content
        self.bn3.results_storage = "dmf:test_models.UserInfo.cluster_1"
        self.assertEqual(self.bn3.full_clean(), None)

    def test_bn_node_validation(self):
        # Test First Step: fields corresponds to Node type
        with self.assertRaises(ValidationError):
            self.mu.deterministic_params = "a, b"
            self.mu.full_clean()
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.node_type = \
                models.BayesianNetworkNode.NODE_TYPE_DETERMINISTIC
            self.mu.full_clean()

        # Test Second Step: Validations on Stochastic Types
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

        # Test Third Step: Validations on Deterministic Types
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

        # Test Fourth Step: Arg parsing
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.distribution_params = '#my_param1, %my_param2'
            self.mu.full_clean()

        # Test Final Step: BayesPy initialization
        self.setUp()
        with self.assertRaises(ValidationError):
            self.mu.distribution_params = "1, 2, 3, 4, 5"
            self.mu.full_clean()

    def test_node_column_validation(self):
        # Node Columns must reference a model
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1_col.ref_model = None
            self.ui_avg1_col.full_clean()
        # Node Columns must be linked to a field or a callable of a model
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1_col.ref_column = None
            self.ui_avg1_col.full_clean()
        # Node Columns must be linked to an existing fields of a model
        self.setUp()
        with self.assertRaises(ValidationError):
            self.ui_avg1_col.ref_column = "non-existant-field"
            self.ui_avg1_col.full_clean()

    def test_bn_get_nodes_names(self):
        expected_output = ['mu', 'tau', 'userinfo.avg1']
        actual_output = list(self.bn1.get_nodes_names())
        self.assertEqual(expected_output, actual_output)

    def test_node_get_data(self):
        # Test no columns assigned
        self.setUp()
        with self.assertRaises(ValueError):
            self.ui_avg1.data_columns.all().delete()
            self.ui_avg1.get_data()
        # Test not-matching column lengths
        self.setUp()
        smaller_data_column, _ = \
            models.BayesianNetworkNodeColumn.objects.get_or_create(
                node=self.ui_avg1,
                ref_model=ContentType.objects.get(
                    model="userinfo2", app_label="test_models"),
                ref_column="avg2"
            )
        with self.assertRaises(ValidationError):
            self.ui_avg1.data_columns.add(smaller_data_column)
            self.ui_avg1.get_data()
        smaller_data_column.delete()
        # Test correct functioning
        self.setUp()
        expected_output = list(test_models.UserInfo.objects.values_list(
            "avg1", flat=True))
        actual_output = list(self.ui_avg1.get_data())
        self.assertEqual(expected_output, actual_output)

    def test_node_get_params_type(self):
        self.assertEqual(self.mu.get_params_type(), "distribution")
        self.assertEqual(self.z.get_params_type(), "deterministic")

    def test_node_reset_engine_object(self):
        self.bn1.perform_inference(recalculate=True)
        self.ui_avg1 = self.bn1.nodes.last()
        self.assertTrue(self.ui_avg1.engine_object is not None)
        self.assertTrue(self.ui_avg1.engine_object_timestamp is not None)
        self.ui_avg1.reset_engine_object()
        self.assertTrue(self.ui_avg1.engine_object is None)
        self.assertTrue(self.ui_avg1.engine_object_timestamp is None)

    def test_node_get_engine_inferred_object(self):
        self.bn1.perform_inference(recalculate=True)
        expected_output = self.bn1.engine_object['userinfo.avg1']
        actual_output = self.ui_avg1.get_engine_inferred_object()
        self.assertEqual(expected_output, actual_output)

    def test_node_resolve_eos_in_params(self):
        self.z.deterministic_params = "x1, x2, x3"
        with self.assertRaises(ValueError):
            self.z.get_engine_object()
        self.z.deterministic_params = "x1, x2, kp=x3"
        with self.assertRaises(ValueError):
            self.z.get_engine_object()

    def test_bn_meta_iterations(self):
        self.setUp()
        self.bn1.engine_meta_iterations = 5
        self.bn1.perform_inference(recalculate=True)
        # There must be a dict of size 5
        self.assertTrue(len(self.bn1._eo_meta_iterations) == 5)
        # containing the same likelihood as there isn't random initialization
        for iteration in self.bn1._eo_meta_iterations:
            self.assertEqual(
                str(self.bn1._eo_meta_iterations[iteration]["L"])[:7],
                "-630.42"
            )

    def test_bn_engine_iterations(self):
        self.setUp()
        self.bn1.engine_iterations = 1
        self.bn1.perform_inference(recalculate=True)
        # There must be a dict of size 1, as engine_meta_iterations defaults to
        # 1
        self.assertTrue(len(self.bn1._eo_meta_iterations) == 1)
        # containing the likelihood of the second iteration
        self.assertTrue(
            str(self.bn1._eo_meta_iterations[0]["eo"].L[0]) != "nan"
        )
        self.assertEqual(
            str(self.bn1._eo_meta_iterations[0]["eo"].L[1]),
            "nan"
        )

    def test_bn_update_eos_struct(self):
        bn1_eos_struct = {n.name: {"dm": n, "eo": None}
                          for n in self.bn1.nodes.all()}
        node = self.bn1.nodes.get(name="userinfo.avg1")
        models.BayesianNetwork.update_eos_struct(bn1_eos_struct, node)
        self.assertTrue('Gamma' in
                        str(bn1_eos_struct['tau']['eo'].__class__))
        self.assertTrue('GaussianARD' in
                        str(bn1_eos_struct['mu']['eo'].__class__))

    def test_node_children(self):
        expected_output = [self.ui_avg1]
        actual_output = list(self.mu.children())
        self.assertEqual(expected_output, actual_output)

    def test_bn_whole_clustering(self):
        self.setUp()
        # Test metadata initialization
        expected_initial_metadata = {
            "clusters_labels": {},
            "prev_clusters_labels": {},
            "clusters_means": {},
            "prev_clusters_means": {},
            "clusters_sizes": {},
            "prev_clusters_sizes": {},
            "columns": [],
        }
        self.assertEqual(self.bn3.metadata, expected_initial_metadata)
        # Test inference and clustering methods through metadata
        self.bn3.perform_inference(recalculate=True)
        expected_metadata = {
            'prev_clusters_labels': {},
            'prev_clusters_means': {},
            'clusters_means': {
                'A': np.array([0., 0.]),
                'B': np.array([16., 16.]),
                'C': np.array([20., 20.]),
                'D': np.array([20., 20.]),
                'E': np.array([25., 25.]),
            },
            'clusters_labels': {'4': 'E', '1': 'A', '5': 'A', '3': 'A',
                                '2': 'B', '8': 'A', '7': 'A', '0': 'C',
                                '6': 'D', '9': 'A'},
            'clusters_sizes': {'A': 0, 'B': 50, 'C': 51, 'D': 49, 'E': 50},
            'columns': ['avg_time_pages', 'avg_time_pages_a']
        }
        output_metadata = self.bn3.metadata
        self.assertEqual(
            output_metadata["prev_clusters_labels"],
            expected_metadata["prev_clusters_labels"]
        )
        self.assertEqual(
            output_metadata["prev_clusters_means"],
            expected_metadata["prev_clusters_means"]
        )
        # Test BN.metadata_update_clusters_sizes()
        self.assertEqual(
            output_metadata["clusters_sizes"],
            expected_metadata["clusters_sizes"]
        )
        # Test BN.assign_clusters_labels()
        for cluster in expected_metadata["clusters_means"]:
            o_cm = output_metadata["clusters_means"][cluster]
            e_cm = expected_metadata["clusters_means"][cluster]
            # Check that the cluster means are 'reasonably close' to
            # the original ones
            self.assertTrue(np.linalg.norm(e_cm - o_cm) ** 2 < 1)
            del(output_metadata["clusters_means"][cluster])
        self.assertEqual(
            output_metadata["clusters_means"],
            {}
        )
        self.assertEqual(
            output_metadata["clusters_labels"],
            expected_metadata["clusters_labels"],
        )
        # Test BN.columns_names_to_metadata()
        self.assertEqual(
            output_metadata["columns"],
            expected_metadata["columns"]
        )
        # Test Results Storage
        # BN.get_results()
        results = self.bn3.get_results()
        # Test resullts are OK (omitting the rest for avoiding pasting a
        # list of size 200)
        self.assertEqual(results[150:], ["B" for x in range(50)])
        # Edge case
        self.assertFalse(self.bn1.get_results())
        # BN.store_results()
        self.bn3.store_results()
        stored_results = test_models.UserInfo.objects.all().values_list(
            'cluster_1', flat=True)
        # Test results are stored OK
        self.assertEqual(list(results), list(stored_results))
        # -> Test BN.threshold_actions validations
        self.bn3.threshold_actions = ":recalculate :not-allowed-action"
        with self.assertRaises(ValidationError):
            self.bn3.full_clean()
        # -> Test BN.counter, BN.counter_threshold and BN.threshold_actions
        # Test Triggering an inference
        self.threshold_actions = ":recalculate"
        prev_timestamp = self.bn3.engine_object_timestamp
        self.bn3.counter = 2
        self.bn3.save()
        # Test the inference has been run by the timestamp
        self.assertTrue(self.bn3.engine_object_timestamp > prev_timestamp)
        # Test the counter was reset
        self.assertEqual(self.bn3.counter, 0)
        # Test BN.assign_cluster()
        self.assertEqual(
            self.bn3.assign_cluster([10, 10]),
            "B"
        )
        self.bn3.reset_inference()
        self.assertFalse(
            self.bn3.assign_cluster([10, 10])
        )
        self.assertFalse(
            self.bn1.assign_cluster([10, 10])
        )

    def test_node_args_parsing(self):
        # Test "general" parsing
        args_string = ('True, :ifr, numpy.ones(2), [[1,2], [3,4]], '
                       'type=rect, sizes=[3, 4,], coords = ([1,2],[3,4]), '
                       'func=numpy.zeros(2), plates=:no')
        expected_output = {
            'args': [
                True,
                ':ifr',
                np.array([1., 1.]),
                    [[1, 2], [3, 4]]
            ],
            'kwargs': {
                'type': 'rect',
                'sizes': [3, 4],
                'coords': ([1, 2], [3, 4]),
                'func': np.array([0., 0.]),
                'plates': ':no',
            }
        }
        output = parse_node_args(args_string)

        # "np.array == np.array" does not return a single bool in NumPy,
        # then the comparison "output == expected_output" does not work
        # with Django tests. I think I also hit a bug, because for some
        # reason, the comparison function that unittest uses for nested
        # lists is the array comparison of NumPy and not the standard list
        # comparison of Python.

        # Test Positional Args
        positions_tested = []
        for position, arg in enumerate(output["args"]):
            # For nested lists, don't know why but it keeps using the
            # NumPy array comparison despites of not being of its class
            if isinstance(arg, np.ndarray) or isinstance(arg, list):
                comp = (expected_output["args"][position] ==
                        output["args"][position])
                if not isinstance(comp, bool):
                    comp = all(comp)
                self.assertEqual(comp, True)
            else:
                self.assertEqual(
                    expected_output["args"][position],
                    output["args"][position]
                )
            positions_tested.insert(0, position)
        # Remove the tested elements from output
        for pt in positions_tested:
            del(output['args'][pt])
        # Test Keyword Args
        for kw in expected_output['kwargs'].keys():
            if (isinstance(expected_output['kwargs'][kw], np.ndarray) or
                    isinstance(expected_output['kwargs'][kw], list)):
                comp = (expected_output['kwargs'][kw] == output["kwargs"][kw])
                if not isinstance(comp, bool):
                    comp = all(comp)
                self.assertEqual(comp, True)
            else:
                self.assertEqual(
                    expected_output['kwargs'][kw],
                    output["kwargs"][kw]
                )
            # Remove the tested element from output
            del(output['kwargs'][kw])
        # Check there is nothing left in the output
        self.assertEqual(output, {"args": [], "kwargs": {}})

        # Test not allowed functions
        with self.assertRaises(ValueError):
            parse_node_args("shutil.rmtree('/')")
        with self.assertRaises(ValueError):
            parse_node_args("eval('<malicious_code>')")

        # Test referencing to a function
        args_string = ('@bayespy.nodes.Gaussian()')
        expected_output = {
            'args': [Gaussian],
            'kwargs': {}
        }
        output = parse_node_args(args_string)
        self.assertEqual(output, expected_output)

        # Test invalid function invocaton
        with self.settings(DJANGO_AI_WHITELISTED_MODULES=["numpy", ]):
            # Reimport the function with the new settings
            from django_ai.bayesian_networks.utils import \
                parse_node_args as pnn
            with self.assertRaises(ValueError):
                pnn("numpy.ones(k)")

        # Test flat output in args parsing
        expected_output = [np.array([1., 1.])]
        output = parse_node_args("numpy.ones(2)", flat=True)
        self.assertTrue(all(output[0] == expected_output[0]))
        output.pop(0)
        self.assertEqual(output, [])

    def test_utils_misc(self):
        # Test Mahalanobis Distance
        self.assertEqual(
            mahalanobis_distance([0, 1], [1, 0], [[2, 0], [0, 2]]), 1.0
        )

    def test_views(self):
        self.setUp()
        # -> Test perform_inference view
        # Test correct inference
        url = reverse('bn_run_inference', args=(self.bn1.id, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.bn1.refresh_from_db()
        self.assertTrue(self.bn1.engine_object is not None)
        # Test incorrect inference
        self.mu.distribution_params = "xxx:bad-dp"
        self.mu.save()
        url = reverse('bn_run_inference', args=(self.bn1.id, ))
        referer = '/admin/bayesian_networks/bayesiannetwork/{}/change/'\
            .format(self.bn1.id)
        response = self.client.get(url, follow=True, HTTP_REFERER=referer)
        self.assertEqual(response.status_code, 200)
        message = list(response.context.get('messages'))[0]
        self.assertEqual(message.tags, "error")
        self.assertTrue("ERROR WHILE PERFORMING INFERENCE" in message.message)

        # -> Test reset_inference view
        url = reverse('bn_reset_inference', args=(self.bn1.id, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.bn1.refresh_from_db()
        self.assertTrue(self.bn1.engine_object is None)

        # -> Test reinitialize_rng view
        state = random.getstate()
        url = reverse('bn_reinitialize_rng')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        new_state = random.getstate()
        self.assertTrue(state is not new_state)

    def tearDown(self):
        self.bn1.image.delete()
        self.mu.image.delete()
        self.tau.image.delete()
        test_models.UserInfo.objects.all().update(cluster_1=None)
