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
from django.template import Context, Template
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from django_ai.ai_base.models import LearningTechnique, DataImputer
from django_ai.ai_base import utils as ai_utils
from django_ai.ai_base.metrics import format_metric

from django_ai.supervised_learning.models import SupervisedLearningTechnique

from .test_models import models as test_models


@modify_settings(
    INSTALLED_APPS={
        'append': 'django_ai.ai_base.tests.test_models',
    })
class TestBase(SimpleTestCase):
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
        # tm_config = apps.get_app_config('test_models')
        # tm_config.models_module = tm_config.models_module or True
        # create_contenttypes(tm_config, verbosity=0)
        # Run migrations
        call_command('migrate', 'test_models', verbosity=0, no_color=True)

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

        cls.statmodel, _ = LearningTechnique.objects\
            .get_or_create(
                name="Test Stat Model",
                data_model="test_models.UserInfo",
                threshold_actions=":recalculate"
            )

    def test_learning_technique__get_data_model(self):
        sm = deepcopy(self.statmodel)
        sm._data_model = None
        self.assertEqual(sm._get_data_model(), test_models.UserInfo)
        # -> Test the cache is working
        self.assertEqual(sm._data_model, test_models.UserInfo)
        self.assertEqual(sm._get_data_model(), test_models.UserInfo)

    def test_learning_technique__get_data_model_attr(self):
        sm = deepcopy(self.statmodel)
        sm._data_model = None
        self.assertEqual(
            sm._get_data_model_attr('learning_fields', []),
            ['avg1', 'cluster_1', 'cluster_2', 'bool_field']
        )
        self.assertEqual(
            sm._get_data_model_attr('non_existant_attr', "NoneValue"),
            "NoneValue"
        )
        self.assertEqual(
            sm._get_data_model_attr('DATA_MODEL_ATTR', "NoneValue"),
            [1, 2, 3]
        )
        sm._data_model = None
        sm.data_model = "test_models.UserInfo2"
        self.assertEqual(
            sm._get_data_model_attr('class_method', "NoneValue"),
            "ClassMethod"
        )

    def test_learning_technique__get_data_learning_fields(self):
        sm = deepcopy(self.statmodel)
        self.assertEqual(
            sm._get_data_learning_fields(),
            ['avg1', 'cluster_1', 'cluster_2', 'bool_field']
        )
        sm.learning_fields = "avg1, sex"
        self.assertEqual(
            sm._get_data_learning_fields(),
            ['avg1', 'sex']
        )
        sm.learning_fields = None
        with mock.patch.object(test_models.UserInfo,
                               'LEARNING_FIELDS', None):
            self.assertEqual(
                sm._get_data_learning_fields(),
                ['age', 'cluster_1', 'cluster_2', 'bool_field']
            )
        sm.data_model = "test_models.UserInfo2"
        sm._data_model = None
        self.assertEqual(
            sm._get_data_learning_fields(),
            []
        )

    def test_learning_technique__get_data_learning_fields_categorical(self):
        sm = deepcopy(self.statmodel)
        self.assertEqual(
            sm._get_data_learning_fields_categorical(),
            ['cluster_1', 'cluster_2', 'bool_field']
        )
        sm.learning_fields_categorical = "sex"
        self.assertEqual(
            sm._get_data_learning_fields_categorical(),
            ['sex']
        )
        sm.learning_fields_categorical = None
        with mock.patch.object(test_models.UserInfo,
                               'LEARNING_FIELDS_CATEGORICAL', None):
            self.assertEqual(
                sm._get_data_learning_fields_categorical(),
                []
            )
        sm.data_model = "test_models.UserInfo2"
        sm._data_model = None
        self.assertEqual(
            sm._get_data_learning_fields_categorical(),
            []
        )

    def test_learning_technique__get_techique(self):
        # Test no childs
        sm = deepcopy(self.statmodel)
        self.assertEqual(sm._get_technique(), sm)
        # Test no technique_fields (should be a technique)
        with mock.patch.object(sm._meta, '_relation_tree', []):
            self.assertEqual(sm._get_technique(), sm)
        # Test return of child
        slt = SupervisedLearningTechnique()
        sm.supervisedlearningtechnique = slt
        self.assertEqual(sm._get_technique(), slt)

    def test_learning_technique__get_categorical_fields_levels(self):
        sm = deepcopy(self.statmodel)
        expected_output = {
            'cluster_1': [None],
            'cluster_2': [0, 1, 2],
            'bool_field': [False, True]
        }
        output = sm._get_categorical_fields_levels()
        self.assertEqual(output, expected_output)
        ui = test_models.UserInfo.objects.last()
        ui.cluster_1 = ""
        ui.save()
        sm._categorical_fields_levels = None
        output = sm._get_categorical_fields_levels()
        self.assertEqual(output, expected_output)
        ui.cluster_1 = None
        ui.save()

    def test_learning_technique__cift_row(self):
        sm = deepcopy(self.statmodel)
        row = [10.5, None, 2, False]
        expected_output = [10.5, 0, 0, 0, 1, 1, 0]
        output = sm._cift_row(row)
        self.assertEqual(output, expected_output)

    def test_learning_technique__cift_reverse_row(self):
        sm = deepcopy(self.statmodel)
        expected_output = [10.5, None, 2, False]
        row = [10.5, 0, 0, 0, 1, 1, 0]
        output = sm._cift_reverse_row(row)
        self.assertEqual(output, expected_output)

    def test_learning_technique__get_categorical_field_indexes(self):
        sm = deepcopy(self.statmodel)
        expected_output = [1, 2, 3]
        output = sm._get_categorical_fields_indexes()
        self.assertEqual(output, expected_output)

    def test_learning_technique__observation_dict_to_list(self):
        sm = deepcopy(self.statmodel)
        obs_dict = {
            "avg1": 11.2, "cluster_1": None, "cluster_2": 1,
            "bool_field": False
        }
        expected_output = [11.2, None, 1, False]
        output = sm._observation_dict_to_list(obs_dict)
        self.assertEqual(output, expected_output)
        sm.SUPPORTS_CATEGORICAL = False
        sm.cift_is_enabled = True
        expected_output = [11.2, 0, 0, 1, 0, 1, 0]
        output = sm._observation_dict_to_list(obs_dict)
        self.assertEqual(output, expected_output)

    def test_learning_technique__observation_list_to_dict(self):
        sm = deepcopy(self.statmodel)
        expected_output = {
            "avg1": 11.2, "cluster_1": None, "cluster_2": 1,
            "bool_field": False
        }
        obs_list = [11.2, None, 1, False]
        output = sm._observation_list_to_dict(obs_list)
        self.assertEqual(output, expected_output)
        sm.SUPPORTS_CATEGORICAL = False
        sm.cift_is_enabled = True
        obs_list = [11.2, 0, 0, 1, 0, 1, 0]
        output = sm._observation_list_to_dict(obs_list)
        self.assertEqual(output, expected_output)

    def test_learning_technique__observation_object_to_list(self):
        sm = deepcopy(self.statmodel)
        obs_obj = test_models.UserInfo(
            avg1=11.2, cluster_1=None, cluster_2=1, bool_field=False
        )
        expected_output = [11.2, None, 1, False]
        output = sm._observation_object_to_list(obs_obj)
        self.assertEqual(output, expected_output)
        sm.SUPPORTS_CATEGORICAL = False
        sm.cift_is_enabled = True
        expected_output = [11.2, 0, 0, 1, 0, 1, 0]
        output = sm._observation_object_to_list(obs_obj)
        self.assertEqual(output, expected_output)

    def test_learning_technique_get_data(self):
        sm = deepcopy(self.statmodel)
        expected_output = [
            list(row) for row in
            test_models.UserInfo.objects.values_list(
                "avg1", "cluster_1", "cluster_2", "bool_field")
        ]
        self.assertTrue(sm.get_data() == expected_output[:-1])
        # -> Test No Categorical Support
        sm._data = None
        sm.SUPPORTS_CATEGORICAL = False
        expected_output = [
            [avg] for avg in
            test_models.UserInfo.objects.values_list("avg1", flat=True)
        ]
        expected_output.pop(-3)
        expected_output.pop(-1)
        self.assertTrue(sm.get_data() == expected_output)
        # -> Test No NA Support
        sm._data = None
        sm.SUPPORTS_CATEGORICAL = False
        sm.SUPPORTS_NA = False
        expected_output = [
            [avg] for avg in
            test_models.UserInfo.objects.values_list("avg1", flat=True)
        ]
        expected_output.pop(-3)
        expected_output.pop(-1)
        self.assertTrue(sm.get_data() == expected_output)
        sm._data = None
        sm.SUPPORTS_CATEGORICAL = True
        sm.SUPPORTS_NA = False
        expected_output = []
        self.assertTrue(sm.get_data() == expected_output)
        # -> Test CIFT
        sm._data = None
        sm.SUPPORTS_CATEGORICAL = False
        sm.SUPPORTS_NA = True
        sm.cift_is_enabled = True
        expected_output = [
            sm._cift_row(list(row)) for row in
            test_models.UserInfo.objects.values_list(
                "avg1", "cluster_1", "cluster_2", "bool_field")
        ]
        self.assertEqual(sm.get_data(), expected_output[:-1])

    def test_learning_technique_resetting(self):
        sm = deepcopy(self.statmodel)
        rotated_metadata = {
            "current": {}, "previous": {"some": "metadata"}
        }
        # -> Test reset_engine_object
        sm.engine_object = {"something": "to_pickle"}
        sm.engine_object_timestamp = timezone.now()
        sm.metadata['inference']['current'] = {"some": "metadata"}
        sm.is_inferred = True
        sm.save()
        sm.engine_object_reset(save=True)
        sm.refresh_from_db()
        self.assertTrue(sm.engine_object is None)
        self.assertTrue(sm.engine_object_timestamp is None)
        self.assertTrue(sm.is_inferred is False)
        self.assertTrue(sm.metadata['inference'] == rotated_metadata)
        sm.engine_object = {"something": "to_pickle"}
        sm.engine_object_timestamp = timezone.now()
        sm.metadata['inference']['current'] = {"some": "metadata"}
        sm.is_inferred = True
        sm.save()
        sm.engine_object_reset(save=False)
        self.assertTrue(sm.engine_object is None)
        self.assertTrue(sm.engine_object_timestamp is None)
        self.assertTrue(sm.is_inferred is False)
        self.assertTrue(sm.metadata["inference"] == rotated_metadata)
        sm.refresh_from_db()
        self.assertTrue(sm.engine_object is not None)
        # -> Test reset_inferece
        sm.engine_object = {"something": "to_pickle"}
        sm.engine_object_timestamp = timezone.now()
        sm.metadata['inference']['current'] = {"some": "metadata"}
        sm.is_inferred = True
        sm.reset_inference(save=True)
        self.assertTrue(sm.engine_object is None)
        self.assertTrue(sm.engine_object_timestamp is None)
        self.assertTrue(sm.is_inferred is False)
        self.assertTrue(sm.metadata["inference"] == rotated_metadata)
        # -> Test reset_metadata
        sm.metadata = {"something": "in_the_metadata"}
        sm.reset_metadata(save=False)
        self.assertTrue(sm.metadata == sm.DEFAULT_METADATA)
        sm.metadata = {}
        sm.save()
        self.assertTrue(
            sm.metadata['inference'] == {'current': {}, 'previous': {}}
        )

    def test_learning_technique_rotate_metadata(self):
        sm = deepcopy(self.statmodel)
        sm.rotate_inference_metadata()
        self.assertTrue(sm.metadata['inference']['current'] == {})
        sm.metadata["inference"]['current'] = {"some": "metadata"}
        sm.rotate_inference_metadata()
        expected_metadata = {
            "current": {}, "previous": {"some": "metadata"}
        }
        self.assertTrue(sm.metadata['inference'] == expected_metadata)

    def test_learning_technique_run_actions(self):
        sm = deepcopy(self.statmodel)
        # -> Test run_actions if_threshold
        sm.counter_threshold = 1
        sm.counter = 1
        with mock.patch.object(sm, 'perform_inference', return_value=True):
            sm.save()
            sm.perform_inference.assert_called_once()
        sm.threshold_actions = ":recalculate :other"
        sm.counter = 1
        with mock.patch.object(sm, 'perform_inference', return_value=True):
            sm.save()
            sm.perform_inference.assert_called_once()
            # :other should be ignored
        sm.counter_threshold = 2
        sm.counter = 1
        with mock.patch.object(sm, 'perform_inference', return_value=True):
            sm.save()
            sm.perform_inference.assert_not_called()
        # -> Test run_actions
        with mock.patch.object(sm, 'perform_inference', return_value=True):
            sm.run_actions()
            sm.perform_inference.assert_called_once()

    def test_learning_technique__get_categorical_mask(self):
        sm = deepcopy(self.statmodel)
        self.assertEqual(sm._get_categorical_mask(), [False, True, True, True])

    def test_learning_technique__get_non_categorical_indexes(self):
        sm = deepcopy(self.statmodel)
        self.assertEqual(sm._get_non_categorical_indexes(), [0])

    def test_learning_technique_get_input_metadata(self):
        sm = deepcopy(self.statmodel)
        expected_metadata = {
            'n_rows': 199,
            'n_cols': 4,
            'cols': ['avg1', 'cluster_1', 'cluster_2', 'bool_field'],
            'cols_na': ['cluster_1'],
            'na_count': {
                'avg1': 1, 'cluster_1': 199, 'cluster_2': 1, 'bool_field': 1,
            },
        }
        metadata = sm.get_learning_data_metadata()
        self.assertEqual(metadata, expected_metadata)

    def test_learning_technique_get_data_model_metadata(self):
        sm = deepcopy(self.statmodel)
        expected_metadata = {
            'learning_fields':
                ['avg1', 'cluster_1', 'cluster_2', 'bool_field'],
            'learning_fields_categorical':
                ['cluster_1', 'cluster_2', 'bool_field'],
            'learning_fields_supported':
                ['avg1', 'cluster_1', 'cluster_2', 'bool_field'],
        }
        metadata = sm.get_data_model_metadata()
        self.assertEqual(metadata, expected_metadata)

    def test_learning_technique_imputer(self):
        sm = deepcopy(self.statmodel)
        sm.SUPPORTS_NA = False
        sm.data_imputer = 'django_ai.ai_base.models.data_imputers.SimpleDataImputer'
        sm.data_imputer_object_reset()
        expected_output = [
            [row[0], 0, row[1], row[2]] for row in
            test_models.UserInfo.objects.values_list("avg1", "cluster_2", "bool_field")
        ]
        expected_output.pop(-1)
        expected_output[-2] = [10.58375562086314, 0, 1, True]
        expected_output[-1] = [14.224856508534689, 0, 1, False]
        output = sm.get_data()
        self.assertEqual(output, expected_output)
        sm = deepcopy(self.statmodel)
        sm.SUPPORTS_NA = False
        sm._data_model = None
        sm.data_model = "test_models.UserInfo2"
        sm.learning_fields = "avg2, avg_time_pages_b"
        sm.data_imputer = 'django_ai.ai_base.models.data_imputers.SimpleDataImputer'
        sm.data_imputer_object_reset()
        expected_output = [
            [row[0], row[1]] for row in
            test_models.UserInfo2.objects.values_list("avg2", "avg_time_pages_b")
        ]
        output = sm.get_data()
        self.assertTrue(output == expected_output)
        # Test no imputer
        sm = deepcopy(self.statmodel)
        sm.SUPPORTS_NA = False
        sm._data_model = None
        sm.data_model = "test_models.UserInfo2"
        sm.learning_fields = "avg2, avg3"
        sm.data_imputer = None
        sm.data_imputer_object_reset()
        expected_output = []
        output = sm.get_data()
        self.assertTrue(output == expected_output)

    def test_learning_technique_impute(self):
        sm = deepcopy(self.statmodel)
        sm.data_imputer = 'django_ai.ai_base.models.data_imputers.SimpleDataImputer'
        expected_output = {'avg1': 10.58375562086314}
        output = sm.impute([None, 0, 1, True])
        self.assertEqual(output, expected_output)
        expected_output = {'bool_field': False}
        output = sm.impute([14.224856508534689, 4, 1, None])
        self.assertEqual(output, expected_output)
        expected_output = {
            'avg1': 10.58375562086314,
            'cluster_2': 1,
            'bool_field': False
        }
        output = sm.impute({'cluster_1': 88})
        self.assertEqual(output, expected_output)
        expected_output = {
            'avg1': 10.58375562086314,
            'cluster_2': 1,
            'bool_field': False,
        }
        output = sm.impute(test_models.UserInfo(cluster_1=0))
        self.assertEqual(output, expected_output)
        sm.data_imputer = None
        expected_output = None
        output = sm.impute(test_models.UserInfo(cluster_1=0))
        self.assertEqual(output, expected_output)

    def test_learning_technique_get_data_imputer_object(self):
        sm = deepcopy(self.statmodel)
        self.assertEqual(sm.get_data_imputer_object(), None)
        sm.SUPPORTS_NA = False
        sm.data_imputer = 'django_ai.ai_base.models.data_imputers.SimpleDataImputer'
        sm.get_data()
        i1 = sm.get_data_imputer_object()
        d1 = sm.get_data()
        i2 = sm.get_data_imputer_object()
        self.assertEqual(i1, i2)
        i3 = sm.get_data_imputer_object(reconstruct=True)
        self.assertNotEqual(i2, i3)
        d2 = sm.get_data()
        self.assertEqual(d1, d2)

    def test_data_imputer__get_imputer(self):
        di = DataImputer()
        self.assertEqual(di._get_imputer(), None)

    def test_learning_technique_model_validation(self):
        sm = deepcopy(self.statmodel)
        # Test threshold actions incorrect content
        with self.assertRaises(ValidationError):
            sm.threshold_actions = \
                ":recalculate :not-supported-kw"
            sm.full_clean()
        # Test threshold actions empty
        sm.threshold_actions = ""
        self.assertEqual(sm.full_clean(), None)
        # Test threshold actions good
        sm.threshold_actions = ":recalculate"
        self.assertEqual(sm.full_clean(), None)
        # -> Test data_model
        sm.data_model = "test_models.UserInfo"
        sm._data_model = None
        self.assertEqual(sm.full_clean(), None)
        # Test incorrect format
        with self.assertRaises(ValidationError):
            sm.data_model = "test_models-UserInfo"
            sm._data_model = None
            sm.full_clean()
        # Test incorrect reference
        with self.assertRaises(ValidationError):
            sm.data_model = "test_models.UserInfo3"
            sm._data_model = None
            sm.full_clean()
        # -> Test data_imputer
        sm = deepcopy(self.statmodel)
        sm.data_imputer = "django_ai.ai_base.models.SimpleDataImputer"
        self.assertEqual(sm.full_clean(), None)
        # Test incorrect format
        with self.assertRaises(ValidationError):
            sm.data_imputer = "django_ai-ai_base-models-SimpleDataImputer"
            sm.full_clean()
        # Test incorrect reference
        with self.assertRaises(ValidationError):
            sm.data_imputer = "django_ai.ai_base.models.OtherImputer"
            sm.full_clean()
        # Test LEARNING_FIELDS
        sm = deepcopy(self.statmodel)
        with self.assertRaises(ValidationError):
            sm.data_model = "test_models.UserInfo2"
            sm._data_model = None
            sm.full_clean()
        with mock.patch.object(test_models.UserInfo,
                               'LEARNING_FIELDS', ['avg1', 'avgX']):
            with self.assertRaises(ValidationError):
                sm.data_model = "test_models.UserInfo"
                sm._data_model = None
                sm.full_clean()
        with mock.patch.object(test_models.UserInfo,
                               'LEARNING_FIELDS', ['avg1', 'avg1']):
            with self.assertRaises(ValidationError):
                sm.data_model = "test_models.UserInfo"
                sm._data_model = None
                sm.full_clean()
        with self.assertRaises(ValidationError):
            sm.learning_fields = 'avg1, avgX'
            sm.full_clean()
        with self.assertRaises(ValidationError):
            sm.learning_fields = 'avg1, avg1'
            sm.full_clean()
        sm.learning_fields = 'avg1, cluster_1, cluster_2'
        sm.learning_fields_categorical = 'cluster_1, cluster_2'
        sm.full_clean()
        # Test LEARNING_FIELDS_CATEGORICAL
        sm = deepcopy(self.statmodel)
        with self.assertRaises(ValidationError):
            sm.data_model = "test_models.UserInfo2"
            sm._data_model = None
            sm.full_clean()
        with mock.patch.object(
                test_models.UserInfo,
                'LEARNING_FIELDS_CATEGORICAL', ['cluster_1', 'avgX']):
            with self.assertRaises(ValidationError):
                sm.data_model = "test_models.UserInfo"
                sm._data_model = None
                sm.full_clean()
        with mock.patch.object(
                test_models.UserInfo,
                'LEARNING_FIELDS_CATEGORICAL', ['cluster_1', 'cluster_1']):
            with self.assertRaises(ValidationError):
                sm.data_model = "test_models.UserInfo"
                sm._data_model = None
                sm.full_clean()
        with mock.patch.object(
                test_models.UserInfo,
                'LEARNING_FIELDS_CATEGORICAL', ['cluster_1', 'cluster_2']):
            sm.data_model = "test_models.UserInfo"
            sm._data_model = None
            sm.full_clean()
        with mock.patch.object(
                test_models.UserInfo,
                'LEARNING_FIELDS_CATEGORICAL', []):
            sm.learning_fields_categorical = None
            sm.full_clean()
        with self.assertRaises(ValidationError):
            sm.learning_fields_categorical = 'cluster_1, avgX'
            sm.full_clean()
        with self.assertRaises(ValidationError):
            sm.learning_fields_categorical = 'cluster_1, cluster_1'
            sm.full_clean()
        sm.learning_fields_categorical = 'cluster_1'
        sm.full_clean()
        sm.learning_fields_categorical = 'cluster_1, cluster_2'
        sm.full_clean()

    def test_templatetags(self):
        # Test getitem filter
        context = Context({
            'mydict': {'key1': 'value1', 'key2': 'value2'}
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{{ mydict|get_item:"key2" }}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('value2', rendered_template)
        # Test items filter
        context = Context({
            'mydict': {'key1': 'value1', 'key2': 'value2'}
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% for key, value in mydict|items %}'
            '{{ key }}:{{ value }}'
            '{% endfor %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('key1:value1', rendered_template)
        self.assertIn('key2:value2', rendered_template)
        # Test is_dict filter
        context = Context({
            'mydict': {'key1': 'value1', 'key2': 'value2'}
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% if mydict|is_dict %}'
            'it\'s a dict!!!'
            '{% endif %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('it\'s a dict!!!', rendered_template)
        context = Context({
            'mydict': ['key1', 'value1', 'key2', 'value2']
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% if mydict|is_dict %}'
            'it\'s a dict!!!'
            '{% endif %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertNotIn('it\'s a dict!!!', rendered_template)
        # Test is_list filter
        context = Context({
            'mydict': {'key1': 'value1', 'key2': 'value2'}
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% if mydict|is_list %}'
            'it\'s a list!!!'
            '{% endif %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertNotIn('it\'s a dict!!!', rendered_template)
        context = Context({
            'mydict': ['key1', 'value1', 'key2', 'value2']
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% if mydict|is_list %}'
            'it\'s a list!!!'
            '{% endif %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('it\'s a list!!!', rendered_template)
        # Test is_dict filter
        context = Context({
            'mydict': {'key1': 'value1', 'key2': 'value2'}
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% if mydict|is_final_dict %}'
            'it\'s a final dict!!!'
            '{% endif %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('it\'s a final dict!!!', rendered_template)
        context = Context({
            'mydict': {'key1': 'value1', 'key2': ['value2', ]}
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% if mydict|is_final_dict %}'
            'it\'s a final dict!!!'
            '{% endif %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertNotIn('it\'s a final dict!!!', rendered_template)
        # Test action_url tag
        context = Context({
            'sm': self.statmodel
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% action_url "perform_inference" sm %}'
            '{% action_url "reset_inference" sm %}'
            '{% action_url "reinitialize_rng" %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn(
            '/django-ai/run-action/perform_inference/learningtechnique/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reset_inference/learningtechnique/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reinitialize_rng',
            rendered_template
        )
        # Test ai_actions tag
        context = Context({
            'original': self.statmodel
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% ai_actions %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn(
            '/django-ai/run-action/perform_inference/learningtechnique/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reset_inference/learningtechnique/1',
            rendered_template
        )
        self.assertIn(
            '/django-ai/run-action/reinitialize_rng',
            rendered_template
        )
        # Test format_inference_metatdata tag
        context = Context({
            'inference_metadata': {"conf": {"eo": {"supports_na": False}}, "o_s": {}},
            'descriptions': {"conf": "Configuration", "eo": "Engine"},
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% format_inference_metadata inference_metadata descriptions %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('Configuration', rendered_template)
        self.assertIn('Engine', rendered_template)
        self.assertIn('O_S', rendered_template)
        # Test format_metatdata_dict tag
        context = Context({
            'inference_metadata': {
                "eo": {
                    "supports_na": {"other": "nested_dict"},
                },
                "final_dict": {"k1": "v1"},
                "other": ["a1, a2"],
                "other2": "a3",
            },
            'descriptions': {"eo": "Engine"},
        })
        template_to_render = Template(
            '{% load django_ai_tags %}'
            '{% format_metadata_dict inference_metadata descriptions %}'
        )
        rendered_template = template_to_render.render(context)
        self.assertIn('Engine', rendered_template)
        self.assertIn('nested_dict', rendered_template)
        self.assertIn('SUPPORTS_NA', rendered_template)
        self.assertIn('<b>k1</b>: v1', rendered_template)
        self.assertIn('a1, a2', rendered_template)
        self.assertIn('a3', rendered_template)

    def test_views(self):
        sm = deepcopy(self.statmodel)
        self.client.login(username='testadmin', password='12345')
        # -> Test RunAction CBV
        # Test correct inference

        def perform_inference_patch(*args, **kwargs):
            sm.engine_object = {"something": "to_pickle"}
            sm.is_inferred = True
            sm.engine_object_timestamp = timezone.now()
            sm.save()
            return(sm.engine_object)

        with mock.patch.object(LearningTechnique,
                               'perform_inference', perform_inference_patch):
            url = reverse('run-action', kwargs={
                "action": "perform_inference",
                "content_type": "learningtechnique",
                "object_id": sm.id, }
            )
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302)
            sm.refresh_from_db()
            self.assertTrue(sm.engine_object is not None)

        # Test incorrect inference
        referer = '/admin/'
        url = reverse('run-action', kwargs={
            "action": "perform_inference", "content_type": "learningtechnique",
            "object_id": sm.id, }
        )
        # For some reason, the request must be done twice, otherwise it won't
        # update the response :\ Maybe it is catching the previous redirect?
        new_response = self.client.get(url, follow=True, HTTP_REFERER=referer)
        new_response = self.client.get(url, follow=True, HTTP_REFERER=referer)
        self.assertEqual(new_response.status_code, 200)
        message = list(new_response.context.get('messages'))[0]
        self.assertEqual(message.tags, "error")
        self.assertTrue("ERROR WHILE PERFORMING INFERENCE" in message.message)

        # -> Test reset_inference action
        url = reverse('run-action', kwargs={
            "action": "reset_inference", "content_type": "learningtechnique",
            "object_id": sm.id, }
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        sm.refresh_from_db()
        self.assertTrue(sm.engine_object is None)

        # -> Test reset_metadata action
        sm.metadata = {"some": "metadata"}
        url = reverse('run-action', kwargs={
            "action": "reset_metadata", "content_type": "learningtechnique",
            "object_id": sm.id, }
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        sm.refresh_from_db()
        self.assertTrue(sm.metadata == sm.DEFAULT_METADATA)

        # -> Test reinitialize_rng action
        state = random.getstate()
        url = reverse('run-action', kwargs={
            "action": "reinitialize_rng", }
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        new_state = random.getstate()
        self.assertTrue(state is not new_state)

        # -> Test unsupported action
        url = reverse('run-action', kwargs={
            "action": "unsupported_action", "content_type": "learningtechnique",
            "object_id": sm.id, }
        )
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 404)

    def test_utils(self):
        statmodel = ai_utils.get_model("ai_base.LearningTechnique")
        self.assertEqual(LearningTechnique, statmodel)

    def test_format_metric(self):
        metric = "test_metric"
        format_string = format_metric(metric, [1])
        self.assertEqual(format_string, 1)
        format_string = format_metric(metric, [])
        self.assertEqual(format_string, None)
        format_string = format_metric(metric, [1, 2, 3])
        self.assertEqual(format_string, '2.000 +/- 1.633')
