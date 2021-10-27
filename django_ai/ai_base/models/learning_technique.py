from django.core.exceptions import ValidationError
from django.db import models
from django.utils.module_loading import import_string
from django.utils.translation import ugettext_lazy as _

from .engine_object import EngineObjectModel
from ..utils import get_model


class LearningTechnique(EngineObjectModel):
    """
    Metaclass for Learning Techniques.

    It defines the common interface so the Techniques can be "plugged"
    along the framework and the applications.
    """
    TYPE_OTHER = 0
    TYPE_SUPERVISED = 1
    TYPE_UNSUPERVISED = 2

    SUPPORTS_CATEGORICAL = True
    SUPPORTS_NA = True

    _categorical_fields_levels = None
    _data = None
    _data_model = None
    _learning_fields_supported = None

    #: Choices for Statistical Model Type
    LT_TYPE_CHOICES = (
        (TYPE_SUPERVISED, _("Supervised")),
        (TYPE_UNSUPERVISED, _("Unsupervised")),
        (TYPE_OTHER, _("Other")),
    )

    #: Unique Name, meant to be used for retrieving the object.
    name = models.CharField(
        _("Name"),
        unique=True,
        max_length=100
    )
    #: Type of the Learning Technique
    technique_type = models.SmallIntegerField(
        _("Learning Technique Type"),
        choices=LT_TYPE_CHOICES, default=TYPE_OTHER,
        blank=True, null=True
    )
    #: Django Model containing the data for the technique in the
    #: "app_label.model" format, i.e. "data_app.Data".
    data_model = models.CharField(
        _("Data Model"),
        max_length=100, default="",
        help_text=(_(
            'Django Model containing the data for the technique in the'
            '"app_label.model" format, i.e. "data_app.Data".'
        ))
    )
    #: Data Model's Fields to be used as input the technique separated
    #: by comma and space, i.e. "avg1, rbc, myfield". If left blank, the
    #: fields defined in LEARNING_FIELDS in the Data Model will be used.
    learning_fields = models.CharField(
        _("Learning Fields"),
        max_length=255, blank=True, null=True,
        help_text=(_(
            'Data Model\'s Fields to be used as input the technique separated'
            'by comma and space, i.e. "avg1, rbc, myfield". If left blank, '
            'the fields defined in LEARNING_FIELDS in the Data Model will be'
            ' used.'
        ))
    )
    #: Django Model containing the data for the technique in the
    #: "app_label.model" format, i.e. "data_app.Data".
    learning_fields_categorical = models.CharField(
        _("Learning Fields Categorical"),
        max_length=255, blank=True, null=True,
        help_text=(_(
            'Subset of Learning Fields that are categorical data separated'
            'by comma and space, i.e. "myfield". If left blank, the fields '
            'defined in LEARNING_FIELDS_CATEGORICAL in the Data Model will '
            'be used.'
        ))
    )
    #: Data Imputer Class to handle possible Non-Available Values in
    #: dotted path format, i.e. "django_ai.imputers.SimpleDataImputer".
    data_imputer = models.CharField(
        _("Data Imputer"),
        max_length=255, blank=True, null=True,
        help_text=(_(
            'Data Imputer Class to handle possible Non-Available Values in '
            'dotted path format, i.e. "django_ai.ai_base.models.SimpleDataImputer".'
        ))
    )
    data_imputer_object_id = models.IntegerField(
        _("Data Imputer Object id"),
        blank=True, null=True,
        help_text=(
            'Data Imputer Object id (internal use)'
        )
    )
    #: Enable Cross Validation (k-Folded)
    cift_is_enabled = models.BooleanField(
        _("Is CIFT Enabled?"),
        default=False,
        help_text=(_(
            'Enable Categorical Indicator Function Transformation (only if '
            'technique does not supports categorical data)'
        ))
    )

    class Meta:
        verbose_name = _("Learning Technique")
        verbose_name_plural = _("Learning Techniques")
        app_label = "ai_base"

    def __str__(self):
        return("[LT] {0}".format(self.name))

    # -> django-ai API
    def data_preprocess(self, data_list):
        """
        Hook for applying transformations to the data (i.e. scaling) before
        it is fed to the Engine.
        """
        return(data_list)

    def get_data(self, fields=None):
        """
        Returns a list with the data constructed either from the data model
        """
        if not self._data:
            self._learning_fields_supported = None
            supported_fields = self._get_data_learning_fields_supported()
            if fields:
                fields_to_retrieve = [f for f in fields if f in supported_fields]
            else:
                fields_to_retrieve = supported_fields
            data_list = [
                list(row) for row in
                self._get_data_queryset().values_list(*fields_to_retrieve)
            ]
            if not self.SUPPORTS_CATEGORICAL and self.cift_is_enabled:
                data_list = [self._cift_row(row) for row in data_list]
            if not self.SUPPORTS_NA and self.data_imputer:
                imputer = self.get_data_imputer_object()
                data_list = [imputer.impute_row(row) for row in data_list]
            data_list = self.data_preprocess(data_list)
            self._data = data_list
        return self._data

    def perform_inference(self, save=True):
        self._data = None
        eo = super().perform_inference(save=save)
        return eo

    def reset_inference(self, save=True):
        self._data = None
        self.data_imputer_object_reset(save=save)
        return super().reset_inference(save=save)

    def get_data_imputer_object(self, reconstruct=False):
        if self.data_imputer:
            if hasattr(self, 'data_imputer_object') and not reconstruct:
                imputer = self.data_imputer_object._get_imputer()
            else:
                if hasattr(self, 'data_imputer_object'):
                    self.data_imputer_object.delete()
                imputer_class = import_string(self.data_imputer)
                imputer = imputer_class.new_imputer_for(technique=self)
                self.data_imputer_object_id = imputer.id
            return imputer
        else:
            return None

    def data_imputer_object_reset(self, save=False):
        if hasattr(self, 'data_imputer_object'):
            self.data_imputer_object.delete()
            self.data_imputer_object = None
        if save:
            self.save()
        return True

    def impute(self, observation):
        if self.data_imputer:
            imputer = self.get_data_imputer_object()
            supported_fields = self._get_data_learning_fields_supported()
            if isinstance(observation, (list, tuple)):
                observation = self._observation_list_to_dict(observation)
            if not isinstance(observation, dict):
                observation = self._observation_object_to_dict(observation)
            imputable_fields = [
                field for field in supported_fields
                if observation.get(field, None) is None and field in supported_fields
            ]
            row = self._observation_dict_to_list(observation)
            imputed_row = imputer.impute_row(row)
            imputed_dict = self._observation_list_to_dict(imputed_row)
            imputed = {
                field: imputed_dict[field] for field in imputable_fields
            }
            return imputed
        else:
            return None

    def get_learning_data_metadata(self):
        n_rows = self._get_data_queryset().count()
        learning_fields_supported = \
            self._get_data_learning_fields_supported()
        na_count = self._get_data_learning_fields_na_count()
        cols_na = self._get_data_learning_fields_na()
        meta = {}
        meta['n_rows'] = n_rows
        meta['n_cols'] = len(learning_fields_supported)
        meta['cols'] = learning_fields_supported
        meta['na_count'] = na_count
        meta['cols_na'] = cols_na
        return meta

    def get_data_model_metadata(self):
        learning_fields = self._get_data_learning_fields()
        learning_fields_categorical = \
            self._get_data_learning_fields_categorical()
        learning_fields_supported = \
            self._get_data_learning_fields_supported()
        meta = {}
        meta['learning_fields'] = learning_fields
        meta['learning_fields_categorical'] = learning_fields_categorical
        meta['learning_fields_supported'] = learning_fields_supported
        return meta

    def get_inference_metadata(self):
        metadata = super().get_inference_metadata()
        metadata["learning_data"] = self.get_learning_data_metadata()
        metadata["scores"] = self.get_inference_scores()
        metadata["conf"]["data_model"] = self.get_data_model_metadata()
        metadata["conf"]["data_imputer"] = self.data_imputer
        metadata["conf"]["eo"]["supports_na"] = self.SUPPORTS_NA
        metadata["conf"]["eo"]["supports_categorical"] = \
            self.SUPPORTS_CATEGORICAL
        metadata["conf"]["eo"]["cift_is_enabled"] = self.cift_is_enabled
        return metadata

    def _get_data_model(self):
        if not self._data_model:
            self._data_model = get_model(self.data_model)
        return self._data_model

    def _get_data_model_attr(self, attr, none_value=None):
        data_model = self._get_data_model()
        attr_value = getattr(data_model, attr.upper(), None)
        if not attr_value:
            attr_get_function_name = '_get_{}'.format(attr.lower())
            attr_get_function = getattr(
                data_model, attr_get_function_name, None
            )
            if attr_get_function:
                if hasattr(attr_get_function, '__func__'):
                    attr_value = attr_get_function()
                else:
                    attr_value = attr_get_function(data_model)
        if attr_value:
            return attr_value
        else:
            return none_value

    def _get_data_learning_fields(self):
        if not self.learning_fields:
            return self._get_data_model_attr('LEARNING_FIELDS', [])
        else:
            return self.learning_fields.split(", ")

    def _get_data_learning_fields_categorical(self):
        if not self.learning_fields_categorical:
            return self._get_data_model_attr('LEARNING_FIELDS_CATEGORICAL', [])
        else:
            if self.learning_fields_categorical == '__none__':
                return []
            else:
                return self.learning_fields_categorical.split(", ")

    def _get_data_learning_fields_supported(self):
        if not self._learning_fields_supported:
            learning_fields = self._get_data_learning_fields()
            learning_fields_categorical = \
                self._get_data_learning_fields_categorical()
            if not self.SUPPORTS_CATEGORICAL and not self.cift_is_enabled:
                supported_fields = [
                    f for f in learning_fields
                    if f not in learning_fields_categorical
                ]
            else:
                supported_fields = learning_fields
            self._learning_fields_supported = supported_fields
        return self._learning_fields_supported

    def _get_data_learning_fields_na_count(self):
        na_count = {}
        learning_fields_supported = self._get_data_learning_fields_supported()
        for col in learning_fields_supported:
            na_count[col] = self._get_data_queryset().filter(
                **{'{}__isnull'.format(col): True}).count()
        return na_count

    def _get_data_learning_fields_na(self):
        n_rows = self._get_data_queryset().count()
        na_count = self._get_data_learning_fields_na_count()
        learning_fields_supported = self._get_data_learning_fields_supported()
        learning_fields_na = [
            f for f in learning_fields_supported if na_count[f] == n_rows
        ]
        return learning_fields_na

    def _get_categorical_mask(self):
        learning_fields = self._get_data_learning_fields()
        learning_fields_categorical = self._get_data_learning_fields_categorical()
        return [f in learning_fields_categorical for f in learning_fields]

    def _get_categorical_fields_indexes(self):
        learning_fields = self._get_data_learning_fields()
        learning_fields_categorical = self._get_data_learning_fields_categorical()
        return [
            learning_fields.index(f) for f in learning_fields
            if f in learning_fields_categorical
        ]

    def _get_non_categorical_indexes(self):
        learning_fields = self._get_data_learning_fields()
        learning_fields_categorical = self._get_data_learning_fields_categorical()
        return [
            learning_fields.index(f) for f in learning_fields
            if f not in learning_fields_categorical
        ]

    def _get_field_name_by_index(self, index, supported=False):
        learning_fields = self._get_data_learning_fields_supported()\
            if supported else self._get_data_learning_fields()
        return learning_fields[index]

    def _get_field_index_by_name(self, name, supported=False):
        learning_fields = self._get_data_learning_fields_supported()\
            if supported else self._get_data_learning_fields()
        return learning_fields.index(name)

    def _observation_dict_to_list(self, observation_dict):
        supported_fields = self._get_data_learning_fields_supported()
        observation_list = []
        for field in supported_fields:
            observation_list.append(observation_dict.get(field, None))
        if not self.SUPPORTS_CATEGORICAL and self.cift_is_enabled:
            observation_list = self._cift_row(observation_list)
        return observation_list

    def _observation_list_to_dict(self, observation_list):
        supported_fields = self._get_data_learning_fields_supported()
        observation_dict = {}
        if not self.SUPPORTS_CATEGORICAL and self.cift_is_enabled:
            observation_list = self._cift_reverse_row(observation_list)
        for index, field in enumerate(supported_fields):
            observation_dict[field] = observation_list[index]
        return observation_dict

    def _observation_object_to_list(self, observation_object):
        supported_fields = self._get_data_learning_fields_supported()
        observation_list = []
        for field in supported_fields:
            observation_list.append(getattr(observation_object, field, None))
        if not self.SUPPORTS_CATEGORICAL and self.cift_is_enabled:
            observation_list = self._cift_row(observation_list)
        return observation_list

    def _observation_object_to_dict(self, observation_object):
        supported_fields = self._get_data_learning_fields_supported()
        observation_dict = {}
        for field in supported_fields:
            observation_dict[field] = getattr(observation_object, field, None)
        return observation_dict

    def _cift_row(self, row):
        new_row = []
        for field_index, field_value in enumerate(row):
            if field_index in self._get_categorical_fields_indexes():
                new_row.extend(self._cift_field(field_index, field_value))
            else:
                new_row.append(field_value)
        return new_row

    def _cift_reverse_row(self, row):
        categorical_levels = self._get_categorical_fields_levels()
        learning_fields = self._get_data_learning_fields()
        learning_fields_categorical = self._get_data_learning_fields_categorical()
        field_lengths = {
            field: len(categorical_levels[field]) if field in learning_fields_categorical else 1
            for field in learning_fields
        }
        new_row = []
        pos = 0
        for field, field_length in field_lengths.items():
            is_categorical = field in learning_fields_categorical
            if is_categorical:
                new_row.append(
                    self._cift_reverse_field(field, row[pos:pos + field_length])
                )
            else:
                new_row.append(row[pos])
            pos += field_length
        return new_row

    def _cift_field(self, field_index, field_value):
        field_name = self._get_field_name_by_index(field_index)
        categorical_levels = self._get_categorical_fields_levels()
        field_levels = categorical_levels[field_name]
        if field_value is not None:
            return [1 if level == field_value else 0 for level in field_levels]
        else:
            return [0 for level in field_levels]

    def _cift_reverse_field(self, field_name, field_list_value):
        categorical_levels = self._get_categorical_fields_levels()
        field_levels = categorical_levels[field_name]
        if 1 in field_list_value:
            level_index = field_list_value.index(1)
        else:
            # NA field
            level_index = 0
        return field_levels[level_index]

    def _get_categorical_fields_levels(self):
        if not self._categorical_fields_levels:
            categorical_levels = {}
            for field in self._get_data_learning_fields_categorical():
                categorical_levels[field] = self._get_field_levels(field)
            self._categorical_fields_levels = categorical_levels
        return self._categorical_fields_levels

    def _get_field_levels(self, field):
        data_model = self._get_data_model()
        django_field = data_model._meta.get_field(field)
        if isinstance(django_field, models.fields.BooleanField):
            levels = [False, True]
        elif django_field.choices:
            levels = [choice for choice, choice_str in django_field.choices]
        else:
            levels = list(self._get_data_queryset()
                          .values_list(field, flat=True).distinct())
            if "" in levels:
                levels.remove("")
        return sorted(levels)

    def _get_data_queryset(self):
        data_model = self._get_data_model()
        learning_fields = self._get_data_learning_fields_supported()
        conds = {
            '{}__isnull'.format(learning_field): True
            for learning_field in learning_fields
        }
        qs = data_model.objects.exclude(**conds)
        if not self.SUPPORTS_NA and not self.data_imputer:
            qs = data_model.objects.all()
            for cond in conds:
                qs = qs.exclude(**{cond: True})
        return qs

    def _get_technique(self):
        techniques_fields = [
            f.related_query_name() for f in self._meta._relation_tree
            if 'ptr_id' in f.attname
        ]
        if techniques_fields:
            for tf in techniques_fields:
                if hasattr(self, tf):
                    return getattr(self, tf)._get_technique()
        return self

    def _get_metadata_descriptions(self):
        descriptions = super()._get_metadata_descriptions()
        descriptions["supports_categorical"] = "Supports Categorical Data"
        descriptions["supports_na"] = "Supports Non-Available Values"
        descriptions["cift_is_enabled"] = "CIFT is Enabled"
        descriptions["data_model"] = "Data Model"
        descriptions["data_imputer"] = "Data Imputer"
        descriptions["learning_fields"] = "Learning Fields"
        descriptions["learning_fields_categorical"] = \
            "Categorical Learning Fields"
        descriptions["learning_fields_supported"] = \
            "Supported Learning Fields"
        descriptions["scores"] = "Scores"
        descriptions["learning_data"] = "Learning Data"
        descriptions["n_rows"] = "Number of Rows"
        descriptions["n_cols"] = "Number of Columns"
        descriptions["cols"] = "Columns"
        descriptions["cols_na"] = "Columns Non-Available"
        descriptions["na_count"] = "Non-Available Data Count"
        data_model = self._get_data_model()
        learning_fields = self._get_data_learning_fields()
        descriptions["learning_fields_descriptions"] = {}
        for field in learning_fields:
            desc = data_model._meta.get_field(field).verbose_name
            descriptions["learning_fields_descriptions"][field] = str(desc)
        return descriptions

    # -> Django Models API
    def clean(self):
        super().clean()
        # Check validity of data_model
        try:
            get_model(self.data_model)
        except Exception:
            raise ValidationError({'data_model': _(
                'Invalid format or reference.'
            )})
        # Check validity of data_imputer
        if self.data_imputer:
            try:
                import_string(self.data_imputer)
            except Exception:
                raise ValidationError({'data_imputer': _(
                    'Invalid format or reference.'
                )})
        # Check validity of learning_fields
        if not self.learning_fields:
            learning_fields = self._get_data_learning_fields()
            if not learning_fields:
                raise ValidationError({'data_model': _(
                    'The Data Model must define learning field either in '
                    'LEARNING_FIELDS or in _get_learning_fields() if they are not '
                    'defined in Technique\'s Learning Fields.'
                )})
            else:
                data_model = self._get_data_model()
                last_field = None
                for field in sorted(learning_fields):
                    if last_field:
                        if field == last_field:
                            raise ValidationError({'data_model': _(
                                'Duplicated field in Learning Fields: {}'
                                .format(field))})
                    else:
                        last_field = field
                    try:
                        getattr(data_model, field)
                    except Exception:
                        raise ValidationError({'data_model': _(
                            'Unrecognized field in Learning Fields: {}'
                            .format(field))})
        else:
            learning_fields = self.learning_fields.split(", ")
            data_model = self._get_data_model()
            last_field = None
            for field in sorted(learning_fields):
                if last_field:
                    if field == last_field:
                        raise ValidationError({'learning_fields': _(
                            'Duplicated field in Learning Fields: {}'
                            .format(field))})
                else:
                    last_field = field
                try:
                    getattr(data_model, field)
                except Exception:
                    raise ValidationError({'learning_fields': _(
                        'Unrecognized field in model {}: {}'
                        .format(self.data_model, field))})
        # Check validity of learning_fields_catgorical
        learning_fields = self._get_data_learning_fields()
        if not self.learning_fields_categorical:
            learning_fields_categorical = \
                self._get_data_learning_fields_categorical()
            if learning_fields_categorical:
                last_field = None
                for field in sorted(learning_fields_categorical):
                    if last_field:
                        if field == last_field:
                            raise ValidationError({'data_model': _(
                                'Duplicated field in Categorical Learning '
                                'Fields: {}'.format(field))})
                    else:
                        last_field = field
                    if field not in learning_fields + ['__none__', ]:
                        raise ValidationError({'data_model': _(
                            'Unrecognized field in Categorical Learning '
                            'Fields: {}'.format(field))})
        else:
            data_model = self._get_data_model()
            learning_fields_categorical = \
                self.learning_fields_categorical.split(", ")
            last_field = None
            for field in sorted(learning_fields_categorical):
                if last_field:
                    if field == last_field:
                        raise ValidationError(
                            {'learning_fields_categorical': _(
                                'Duplicated field in Learning Fields '
                                'Categorical: {}'.format(field))})
                else:
                    last_field = field
                if field not in learning_fields + ['__none__', ]:
                    raise ValidationError({'learning_fields_categorical': _(
                        'Unrecognized field for model {} Learning Fields: '
                        '{}'.format(self.data_model, field))})

    def save(self, *args, **kwargs):
        """
        Base save() processing
        """
        super().save(*args, **kwargs)
