#
"""
Tests for `django-ai` `ai-base` module.
====================================
"""
import random
import numpy as np
from copy import deepcopy
from unittest import mock

from django.core.management import call_command
from django.core.exceptions import ValidationError
from django.test import Client, SimpleTestCase, modify_settings
from django.contrib.auth.models import User

from django_ai.supervised_learning.metrics import format_metric
from django_ai.supervised_learning.models import (
    HGBTreeClassifier, HGBTreeRegressor,
    SupervisedLearningTechnique, SupervisedLearningImputer,
    SVC, SVR
)

from .sl_test_models import models as sl_test_models


@modify_settings(
    INSTALLED_APPS={
        'append': 'django_ai.supervised_learning.tests.sl_test_models',
    })
class TestSupervisedLearning(SimpleTestCase):
    """
    SimpleTestCase is needed otherwise migrations won't run inside transactions in SQLite
    in Django 3.1
    """
    databases = "__all__"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Run migrations for test models
        call_command('migrate', 'sl_test_models', verbosity=0, no_color=True)

        # Set the seeds
        random.seed(123456)
        np.random.seed(123456)

        # Set up the user
        cls.user, _ = User.objects.get_or_create(
            username='testadmin', email='testadmin@example.com',
            is_superuser=True, is_staff=True
        )
        cls.user.set_password("12345")
        cls.user.save()
        cls.client = Client()
        cls.client.login(username='testadmin', password='12345')

        cls.slt, _ = SupervisedLearningTechnique.objects\
            .get_or_create(
                name="Test Supervised Learning Technique",
                data_model="sl_test_models.UserInfo",
            )

        cls.hgb, _ = HGBTreeClassifier.objects.get_or_create(
            name="Test HGBTree",
            data_model="sl_test_models.UserInfo",
            max_iter=10, verbose=False
        )

        cls.hgb_r, _ = HGBTreeRegressor.objects.get_or_create(
            name="Test HGBTree - R",
            data_model="sl_test_models.UserInfo",
            max_iter=10, verbose=False
        )

        cls.svm, _ = SVC.objects.get_or_create(
            name="Test SVM",
            data_model="sl_test_models.UserInfo",
            data_imputer="django_ai.ai_base.models.data_imputers.SimpleDataImputer",
            cift_is_enabled=True,
            max_iter=None
        )

        cls.svr, _ = SVR.objects.get_or_create(
            name="Test SVM - R",
            data_model="sl_test_models.UserInfo",
            data_imputer="django_ai.ai_base.models.data_imputers.SimpleDataImputer",
            cift_is_enabled=True,
            max_iter=None
        )

        cls.sli, _ = SupervisedLearningImputer.objects.get_or_create(
            name="Test Supervised Learning Imputer",
            data_model="sl_test_models.UserInfo",
            cift_is_enabled=True,
        )

        cls.classifiers = [cls.svm, cls.hgb, ]

        cls.regressors = [cls.svr, cls.hgb_r, ]

        cls.slts_introspection = [cls.svm, cls.svr, cls.hgb, cls.hgb_r, ]

        cls.imputers = [
            'django_ai.supervised_learning.models.SVMImputer',
            'django_ai.supervised_learning.models.HGBTreeImputer',
        ]

    def test_get_targets(self):
        slt = deepcopy(self.slt)
        d1 = slt.get_targets()
        expected_output = \
            [label for label in
             sl_test_models.UserInfo.objects.values_list("sex", flat=True)]
        self.assertEqual(d1, expected_output[:-1])

    def test__get_cv_metrics(self):
        slt = deepcopy(self.slt)
        self.assertEqual(slt._get_cv_metrics(), [])
        slt.cv_metrics = "accuracy, precision"
        self.assertEqual(slt._get_cv_metrics(), ["accuracy", "precision"])

    def test_perform_cross_validation(self):
        h1 = deepcopy(self.hgb)
        h1.cv_is_enabled = True
        h1.cv_folds = 2
        h1.cv_metrics = 'accuracy'
        h1.perform_inference(save=False)
        scores = h1.perform_cross_validation()
        self.assertEqual(
            scores, h1.metadata['inference']['current']['scores']['cv']
        )
        h1.cv_is_enabled = False
        h1.perform_inference()
        self.assertTrue(
            'cv' not in h1.metadata['inference']['current']['scores']
        )

    def test_predict(self):
        slt = deepcopy(self.slt)
        slt.is_inferred = False
        p1 = slt.predict([[2, 3]], include_scores=True)
        self.assertTrue(p1 is None)
        slt.is_inferred = True
        slt.SUPPORTS_NA = False
        with mock.patch.object(slt,
                               'engine_object_predict', return_value=[[0]]):
            p0 = slt.predict([1, None])
            self.assertTrue(p0 == [[0]])

    def test__get_monotonic_constraints_list(self):
        slt = deepcopy(self.slt)
        slt.monotonic_constraints = "None"
        m_cs = slt._get_monotonic_constraints_list()
        self.assertEqual(m_cs, [0, 0, 0, 0])
        slt.monotonic_constraints = "avg1: 1"
        m_cs = slt._get_monotonic_constraints_list()
        self.assertEqual(m_cs, [1, 0, 0, 0])
        slt.data_model = "sl_test_models.UserInfo2"
        slt._data_model = None
        slt.learning_fields = "avg2"
        slt.monotonic_constraints = None
        m_cs = slt._get_monotonic_constraints_list()
        self.assertEqual(m_cs, [0])

    def test_through_classifiers(self):
        classifiers = deepcopy(self.classifiers)
        for classifier in classifiers:
            classifier.cv_is_enabled = True
            classifier.cv_folds = 2
            classifier.cv_metrics = 'accuracy'
            eo1 = classifier.perform_inference()
            p1, s1 = classifier.predict(
                [{'avg1': 1.2, 'cluster_1': None}],
                include_scores=True
            )
            p2 = classifier.predict([[2.3, None]], include_scores=False)
            eo2 = classifier.perform_inference(save=False)
            p3, s3 = classifier.predict(
                [{'avg1': 1.2, 'cluster_1': None}],
                include_scores=True
            )
            ui = sl_test_models.UserInfo.objects.first()
            p4 = classifier.predict([ui], include_scores=False)
            self.assertEqual(p3, p1)
            self.assertTrue((abs(s1[0] - s3[0]) < 0.1))
            self.assertEqual(p4, p2)
            self.assertNotEqual(eo1, eo2)

    def test_svm(self):
        svm = deepcopy(self.svm)
        svm.cv_is_enabled = True
        svm.cv_folds = 2
        svm.cv_metrics = 'accuracy'
        svm.max_iter = 10000
        svm.estimate_probability = False
        svm.perform_inference()
        svm.predict([[1, 2]], include_scores=True)
        svm.predict([[1, None]], include_scores=True)
        svm.predict([[1, None, None, None]], include_scores=True)
        svm.predict([sl_test_models.UserInfo.objects.last()])

    def test_through_regressors(self):
        regressors = deepcopy(self.regressors)
        for regressor in regressors:
            regressor.learning_target = 'avg1'
            regressor.learning_fields = 'cluster_1, cluster_2, bool_field'
            regressor.cift_is_enabled = True
            regressor.cv_is_enabled = True
            regressor.cv_folds = 2
            regressor.cv_metrics = 'explained_variance'
            eo1 = regressor.perform_inference()
            p1, s1 = regressor.predict(
                [{'cluster_1': None, 'cluster_2': 1, 'bool_field': True}],
                include_scores=True
            )
            p2 = regressor.predict([[None, 2, True]], include_scores=False)
            eo2 = regressor.perform_inference(save=False)
            p3, s3 = regressor.predict(
                [{'cluster_1': None, 'cluster_2': 1, 'bool_field': True}],
                include_scores=True
            )
            ui = sl_test_models.UserInfo.objects.first()
            p4 = regressor.predict([ui], include_scores=False)
            self.assertEqual(p3, p1)
            self.assertTrue(s1, s3)
            self.assertEqual(p4, p2)
            self.assertNotEqual(eo1, eo2)

    def test_svr(self):
        svr = deepcopy(self.svr)
        svr.learning_target = 'avg1'
        svr.learning_fields = 'cluster_1, cluster_2, bool_field'
        svr.cift_is_enabled = True
        svr.cv_is_enabled = True
        svr.cv_folds = 2
        svr.cv_metrics = 'explained_variance'
        svr.max_iter = None
        svr.perform_inference()
        svr.predict([[1, 2]], include_scores=True)
        svr.predict([[1, None]], include_scores=True)
        svr.predict([[1, 1, False]], include_scores=True)
        svr.predict([sl_test_models.UserInfo.objects.last()])

    def test_through_imputers(self):
        imputers = deepcopy(self.imputers)
        for imputer in imputers:
            slt = deepcopy(self.slt)
            slt.data_imputer = imputer
            slt.data_imputer_object_reset()
            slt.get_data()
            di = slt.get_data_imputer_object()
            di.perform_inference(save=False)
            slt.impute([10, 0, 1, True])
            slt.impute([None, 0, 1, True])
            slt.impute(
                sl_test_models.UserInfo(avg1=14.224856508534689, cluster_2=1)
            )
            slt.impute(sl_test_models.UserInfo(bool_field=False))
            with self.assertRaises(ValueError):
                slt.impute({'cluster_1': 88})

    def test_supervised_learning_imputer(self):
        sli = deepcopy(self.sli)
        self.assertEqual(sli._get_technique_for_regression(), None)
        self.assertEqual(sli._get_technique_for_classification(), None)
        sli.classification_technique = \
            'django_ai.supervised_learning.models.svm.SVC'
        sli.regression_technique = \
            'django_ai.supervised_learning.models.svm.SVR'
        self.assertEqual(sli._get_technique_for_regression(), SVR)
        self.assertEqual(sli._get_technique_for_classification(), SVC)
        # Test cache
        self.assertEqual(sli._get_technique_for_regression(), SVR)
        self.assertEqual(sli._get_technique_for_classification(), SVC)

    def test_svm_imputer(self):
        slt = deepcopy(self.slt)
        slt.data_imputer = 'django_ai.supervised_learning.models.SVMImputer'
        expected_output = {}
        output = slt.impute([10, 0, 1, True])
        self.assertEqual(output, expected_output)
        expected_output = {'avg1': 9.864762218296288}
        output = slt.impute([None, 0, 1, True])
        self.assertEqual(output, expected_output)
        expected_output = {
            'bool_field': False,
            'cluster_1': 0
        }
        output = slt.impute(
            sl_test_models.UserInfo(
                avg1=14.224856508534689, cluster_2=1
            )
        )
        self.assertEqual(output, expected_output)
        expected_output = {
            'avg1': 11.02720541243457,
            'cluster_1': 0,
            'cluster_2': 1,
        }
        output = slt.impute(
            sl_test_models.UserInfo(bool_field=False)
        )
        self.assertEqual(output, expected_output)
        with self.assertRaises(ValueError):
            output = slt.impute({'cluster_1': 88})

    def test_introspection(self):
        slts = deepcopy(self.slts_introspection)
        for slt in slts:
            slt.data_imputer = \
                'django_ai.supervised_learning.models.IntrospectionImputer'
            slt.data_imputer_object_reset()
            di = slt.get_data_imputer_object()
            di.perform_inference(save=False)
            slt.get_data()
            di = slt.get_data_imputer_object()
            di.perform_inference(save=False)
            slt.impute([10, 0, 1, True])
            slt.impute([None, 0, 1, True])
            slt.impute(
                sl_test_models.UserInfo(avg1=14.224856508534689, cluster_2=1)
            )
            slt.impute(sl_test_models.UserInfo(bool_field=False))
        with self.assertRaises(ValidationError):
            slt = deepcopy(self.slt)
            slt.data_imputer = \
                'django_ai.supervised_learning.models.IntrospectionImputer'
            slt.impute([None, 0, 1, False])

    def test_sl_validation(self):
        slt = deepcopy(self.slt)
        # -> Test Learning Labels
        slt.learning_target = 'avgX'
        with self.assertRaises(ValidationError):
            slt.full_clean()
        with mock.patch.object(sl_test_models.UserInfo,
                               'LEARNING_TARGET', 'avgX'):
            slt.learning_target = None
            with self.assertRaises(ValidationError):
                slt.full_clean()
        with mock.patch.object(sl_test_models.UserInfo,
                               'LEARNING_TARGET', None):
            slt.learning_target = None
            with self.assertRaises(ValidationError):
                slt.full_clean()
        with mock.patch.object(sl_test_models.UserInfo,
                               'LEARNING_TARGET', None):
            slt.data_model = "sl_test_models.UserInfo"
            slt.learning_target = "avg1"
            self.assertEqual(slt.full_clean(), None)
        # -> Test Monotonic Constraints
        slt.monotonic_constraints = 'avgXfds'
        with self.assertRaises(ValidationError):
            slt.full_clean()
        slt.monotonic_constraints = 'avg3: 3'
        with self.assertRaises(ValidationError):
            slt.full_clean()
        slt.monotonic_constraints = 'avg1: 3'
        with self.assertRaises(ValidationError):
            slt.full_clean()
        slt.monotonic_constraints = 'avg1: -1'
        self.assertEqual(slt.full_clean(), None)
        slt.monotonic_constraints = 'None'
        self.assertEqual(slt.full_clean(), None)
        slt.monotonic_constraints = None
        with mock.patch.object(
                sl_test_models.UserInfo,
                'LEARNING_FIELDS_MONOTONIC_CONSTRAINTS', None):
            self.assertEqual(slt.full_clean(), None)
        with mock.patch.object(
                sl_test_models.UserInfo,
                'LEARNING_FIELDS_MONOTONIC_CONSTRAINTS', 'None'):
            self.assertEqual(slt.full_clean(), None)
        with mock.patch.object(
                sl_test_models.UserInfo,
                'LEARNING_FIELDS_MONOTONIC_CONSTRAINTS', "avgX: 1"):
            with self.assertRaises(ValidationError):
                slt.full_clean()
        with mock.patch.object(
                sl_test_models.UserInfo,
                'LEARNING_FIELDS_MONOTONIC_CONSTRAINTS', "avg1: -3"):
            with self.assertRaises(ValidationError):
                slt.full_clean()
        # -> Test CV validation
        # Test missing folds and metric
        with self.assertRaises(ValidationError):
            slt.cv_is_enabled = True
            slt.full_clean()
        # Test missing metric
        with self.assertRaises(ValidationError):
            slt.cv_is_enabled = True
            slt.cv_folds = 5
            slt.cv_metrics = None
            slt.full_clean()
        # Test missing fold
        with self.assertRaises(ValidationError):
            slt.cv_is_enabled = True
            slt.cv_folds = None
            slt.cv_metrics = 'accuracy'
            slt.full_clean()
        # Test wrong metrics
        with self.assertRaises(ValidationError):
            slt.cv_is_enabled = True
            slt.cv_folds = 5
            slt.cv_metrics = 'accccuracy, non-rec-metric'
            slt.full_clean()
        # Test correct content
        slt.cv_is_enabled = True
        slt.cv_folds = 5
        slt.cv_metrics = 'accuracy'
        self.assertEqual(slt.full_clean(), None)

    def test_format_metric(self):
        metric = "test_metric"
        format_string = format_metric(metric, [1])
        self.assertEqual(format_string, 1)
        format_string = format_metric(metric, [])
        self.assertEqual(format_string, None)
        format_string = format_metric(metric, [1, 2, 3])
        self.assertEqual(format_string, '2.000 +/- 1.633')
