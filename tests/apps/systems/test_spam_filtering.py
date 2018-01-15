#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_spam_filtering.py
------------

Tests for `django-ai.systems`\ 's `spam_filtering` module.
"""
import random
import numpy as np

from django.test import (TestCase, )
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import (ValidationError, ImproperlyConfigured)

from django_ai.base.models import (DataColumn, )
from django_ai.supervised_learning.models.svm import (SVC, )
from django_ai.systems.spam_filtering.models import (SpamFilter, )
from tests.test_models import models as test_models


class TestSpamFiltering(TestCase):

    def setUp(self):
        # Set the seeds
        random.seed(123456)
        np.random.seed(123456)
        self.pretraining_data = list(test_models.MySFPT.objects.values_list(
            "content", flat=True
        ))
        self.pretraining_labels = list(test_models.MySFPT.objects.values_list(
            "is_spam", flat=True
        ))
        self.spam_data = list(test_models.SpammableModel.objects.values_list(
            "comment", flat=True
        ))
        self.spam_labels = list(test_models.SpammableModel.objects.values_list(
            "is_spam", flat=True
        ))
        self.svm_sf, _ = SVC.objects.get_or_create(
            name="SVM for SF tests",
            penalty_parameter=0.1,
            kernel="linear",
        )
        self.spam_filter_1, _ = SpamFilter.objects.get_or_create(
            name="Spam Filter for tests",
            classifier="supervised_learning.SVC|SVM for SF tests",
            spam_model_is_enabled=True,
            spam_model_model="test_models.SpammableModel",
            pretraining="test_models.MySFPT",
            cv_folds=2,
        )
        self.comment_dc, _ = DataColumn.objects.get_or_create(
            content_type=ContentType.objects.get_for_model(SpamFilter),
            object_id=self.spam_filter_1.id,
            ref_model=ContentType.objects.get(
                app_label="test_models", model="spammablemodel"),
            ref_column="comment",
            position=0,
        )
        self.spammable_model_1, _ = test_models.SpammableModel.objects\
            .get_or_create(
                comment="Buy a Gorilla Online!"
            )

    def test_spam_filter_validation(self):
        self.setUp()
        # -> Test validaton on the base case
        self.assertEqual(self.spam_filter_1.full_clean(), None)
        # -> Test invalid classifier cases
        self.setUp()
        # Test invalid format
        self.spam_filter_1.classifier = \
            "supervised_learning.SVC.SVM for SF tests"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()
        # Test invalid app and model
        self.spam_filter_1.classifier = \
            "sup_learn.SVD|SVM for SF tests"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()
        # Test invalid model object
        self.spam_filter_1.classifier = \
            "supervised_learning.SVC|Non-existant SVM"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()
        # -> Test invalid pretraining cases
        self.setUp()
        # Test invalid format
        self.spam_filter_1.pretraining = "test_models|MySFPT"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()
        # Test invalid app and models
        self.spam_filter_1.pretraining = "testo_models.YourSFPT"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()
        # -> Test invalid spammable model cases
        self.setUp()
        # Test invalid format
        self.spam_filter_1.spam_model_model = "test_models|SpammableModel"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()
        # Test invalid app and models
        self.spam_filter_1.spam_model_model = "testo_models.NonSpammable"
        with self.assertRaises(ValidationError):
            self.spam_filter_1.full_clean()

    def test_spam_filter_save(self):
        self.spam_filter_1.metadata = {}
        self.spam_filter_1.save()
        empty_metadata = {"current_inference": {},
                          "previous_inference": {}}
        self.assertEqual(self.spam_filter_1.metadata, empty_metadata)

    def test_spam_filter_pretraining(self):
        self.setUp()
        # -> Test regular usage
        self.assertEqual(self.spam_filter_1.get_pretraining_data(),
                         self.pretraining_data)
        self.assertEqual(self.spam_filter_1.get_pretraining_labels(),
                         self.pretraining_labels)
        self.assertEqual(self.spam_filter_1.get_data(),
                         self.spam_data + self.pretraining_data)
        self.assertEqual(self.spam_filter_1.get_labels(),
                         self.spam_labels + self.pretraining_labels)
        # -> Disable pre-training
        self.spam_filter_1.pretraining = None
        self.assertEqual(self.spam_filter_1.get_pretraining_data(), None)
        self.assertEqual(self.spam_filter_1.get_pretraining_labels(), None)
        self.assertEqual(self.spam_filter_1.get_data(), self.spam_data)
        self.assertEqual(self.spam_filter_1.get_labels(), self.spam_labels)

    def test_spam_filter_use_spammable_model(self):
        self.setUp()
        # -> Test disable spam model (enabled is tested in the base case)
        self.spam_filter_1.spam_model_is_enabled = False
        self.spam_filter_1.labels_column = "test_models.SpammableModel.is_spam"
        self.assertEqual(self.spam_filter_1.get_data(),
                         self.spam_data + self.pretraining_data)
        self.assertEqual(self.spam_filter_1.get_labels(),
                         self.spam_labels + self.pretraining_labels)

    def test_spam_filter_engine_object_vectorizer(self):
        self.setUp()
        # -> Test CountVectorizer (TF-IDF is tested in the base case)
        self.spam_filter_1.bow_use_tf_idf = False
        eov = self.spam_filter_1.get_engine_object_vectorizer(reconstruct=True)
        self.assertTrue("CountVectorizer" in str(eov.__class__))
        self.spam_filter_1.perform_inference(recalculate=True)
        eov_cached = self.spam_filter_1.get_engine_object_vectorizer()
        self.assertEqual(eov.vocabulary_, eov_cached.vocabulary_)

    def test_spam_filter_engine_object_data(self):
        self.setUp()
        eod = self.spam_filter_1.get_engine_object_data()
        eod_cached = self.spam_filter_1.get_engine_object_data()
        # import ipdb; ipdb.set_trace()
        self.assertTrue((eod - eod_cached).nnz == 0)

    def test_spam_filter_inference(self):
        self.setUp()
        # -> Test "regular" inference
        self.spam_filter_1.reset_inference()
        # If the resulting metadata is the expected one, then all the process
        # went fine
        expected_metadata = {
            'previous_inference': {},
            'current_inference': {
                'cv': {
                    '2std': 0.0,
                    'mean': 1.0,
                    'conf': {
                        'metric': None,
                        'folds': 2
                    },
                    # Numpy arrays need to be compared differently
                    # 'scores': np.array([ 1.,  1.])
                },
                'bow_is_enabled': True,
                'vectorizer_conf': {
                    'bow_use_tf_idf': True,
                    'binary': False,
                    'bow_is_enabled': True,
                    'df_min_max': '1.0 / 1.0',
                    'analyzer': 'Word',
                    'ngram_range': '(1, 1)',
                    'str': ('BoW Representation: (TF-IDF Transformation) '
                            'Analyzer: Word (1, 1) - Min / Max DF: 1.0 / 1.0')
                },
                'input_dimensionality': (10, 20),
                'classifier_conf': {
                    'kernel': 'Linear',
                    'kernel_details': '',
                    'str': 'Kernel: Linear, Penalty: 0.1',
                    'name': 'SVM for SF tests',
                    'penalty_parameter': 0.1
                }
            }
        }
        em_cv_scores = np.array([1., 1.])
        self.spam_filter_1.perform_inference(recalculate=True)
        self.assertEqual(self.spam_filter_1.is_inferred, True)
        actual_metadata = self.spam_filter_1.metadata
        am_cv_scores = actual_metadata["current_inference"]["cv"].pop("scores")
        self.assertEqual(actual_metadata, expected_metadata)
        self.assertTrue(all(em_cv_scores == am_cv_scores))
        # -> Test unicode representation
        self.spam_filter_1.bow_is_enabled = False
        expected_metadata = {
            'previous_inference': {
                'cv': {
                    '2std': 0.0,
                    'mean': 1.0,
                    'conf': {
                        'metric': None,
                        'folds': 2
                    }
                },
                'bow_is_enabled': True,
                'vectorizer_conf': {
                    'bow_use_tf_idf': True,
                    'binary': False,
                    'bow_is_enabled': True,
                    'df_min_max': '1.0 / 1.0',
                    'analyzer': 'Word',
                    'ngram_range': '(1, 1)',
                    'str': ('BoW Representation: (TF-IDF Transformation) '
                            'Analyzer: Word (1, 1) - Min / Max DF: 1.0 / 1.0')
                },
                'input_dimensionality': (10, 20),
                'classifier_conf': {
                    'kernel': 'Linear',
                    'kernel_details': '',
                    'str': 'Kernel: Linear, Penalty: 0.1',
                    'name': 'SVM for SF tests',
                    'penalty_parameter': 0.1
                }
            },
            'current_inference': {
                'cv': {
                    '2std': 0.5,
                    'mean': 0.75,
                    'conf': {
                        'metric': None,
                        'folds': 2
                    },
                    # Numpy arrays need to be compared differently
                    # 'scores': array([ 0.5,  1. ])
                },
                'bow_is_enabled': False,
                'vectorizer_conf': {
                    'bow_use_tf_idf': True,
                    'binary': False,
                    'bow_is_enabled': False,
                    'df_min_max': '1.0 / 1.0',
                    'analyzer': 'Word',
                    'ngram_range': '(1, 1)',
                    'str': 'UTF-8 Representation (Vectorizer not enabled)'
                },
                'input_dimensionality': (10, 25),
                'classifier_conf': {
                    'kernel': 'Linear',
                    'kernel_details': '',
                    'str': 'Kernel: Linear, Penalty: 0.1',
                    'name': 'SVM for SF tests',
                    'penalty_parameter': 0.1
                }
            }
        }
        em_cv_scores = np.array([0.5, 1.])
        self.spam_filter_1.perform_inference(recalculate=True)
        actual_metadata = self.spam_filter_1.metadata
        am_cv_scores = actual_metadata["current_inference"]["cv"].pop("scores")
        self.assertEqual(actual_metadata, expected_metadata)
        self.assertTrue(all(em_cv_scores == am_cv_scores))

    def test_spam_filter_predict(self):
        self.setUp()
        self.spam_filter_1.reset_inference()
        # -> Test non-inferred spam filter
        self.assertEqual(self.spam_filter_1.predict(["Buy a Whale Online!"]),
                         None)
        # -> Predict using BOW
        self.spam_filter_1.bow_is_enabled = True
        self.spam_filter_1.perform_inference(recalculate=True)
        self.assertEqual(self.spam_filter_1.predict(["Buy a Panda Online!"]),
                         [True])
        # -> Predict using Unicode representation
        self.spam_filter_1.bow_is_enabled = False
        self.spam_filter_1.perform_inference(recalculate=True)
        self.assertEqual(self.spam_filter_1.predict(["Buy a Rhino Online!"]),
                         [True])

    def test_spam_filter_cv(self):
        self.setUp()
        # -> Test using BOW
        expected_scores = np.array([1., 1.])
        actual_scores = self.spam_filter_1.perform_cross_validation()
        self.assertTrue(all(expected_scores == actual_scores))
        # -> Test using Unicode representation
        self.spam_filter_1.bow_is_enabled = False
        expected_scores = np.array([0.5, 1.])
        actual_scores = self.spam_filter_1.perform_cross_validation()
        self.assertTrue(all(expected_scores == actual_scores))

    def test_spammable_model_save(self):
        self.setUp()
        # -> Test Non-existant Spam Filter Model
        self.spammable_model_1.SPAM_FILTER = "Non-existant Spam Filter"
        with self.assertRaises(ImproperlyConfigured):
            self.spammable_model_1.save()
        # -> Test Non-existant Spammable Field
        self.setUp()
        self.spammable_model_1.SPAMMABLE_FIELD = "non_existant_field"
        with self.assertRaises(ImproperlyConfigured):
            self.spammable_model_1.save()
        # -> Test correct saving
        self.setUp()
        self.spam_filter_1.perform_inference(recalculate=True, save=True)
        self.assertEqual(self.spammable_model_1.save(), None)

    def test_vect_conf(self):
        self.setUp()
        self.spam_filter_1.bow_binary = True
        vcdict = self.spam_filter_1.get_vect_conf_dict()
        self.assertTrue("Binary" in vcdict['str'])
