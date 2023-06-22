from django.db import models
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _

from django_ai.ai_base.models import DataImputer
from django_ai.ai_base.utils import allNotNone


class SupervisedLearningImputer(DataImputer):
    NA_COLUMN_FILL = 0

    _classification_technique = None
    _regression_technique = None
    _non_na_fields = None

    classification_technique = models.CharField(
        _("Classification Technique"),
        max_length=255,
        blank=True,
        null=True,
        help_text=(
            _(
                "Supervised Learning Technique Class to handle categorical field "
                "imputing."
            )
        ),
    )
    regression_technique = models.CharField(
        _("Regression Technique"),
        max_length=255,
        blank=True,
        null=True,
        help_text=(
            _(
                "Supervised Learning Technique Class to handle non-categorical "
                "field imputing."
            )
        ),
    )

    def engine_object_init(self):
        field_imputer = {}
        learning_fields_categorical = (
            self._get_data_learning_fields_categorical()
        )
        for field in self.get_non_na_fields():
            if field in learning_fields_categorical:
                learner = self._get_technique_for_classification()
            else:
                learner = self._get_technique_for_regression()
            field_imputer[field] = learner(
                data_model=self.data_model,
                learning_target=field,
                cift_is_enabled=True,
            )
        return field_imputer

    def engine_object_perform_inference(self, field, expl_fields):
        learning_fields_categorical = (
            self._get_data_learning_fields_categorical()
        )
        eo = self.get_engine_object()
        eo[field].learning_fields = ", ".join(expl_fields)
        eo[field].learning_fields_categorical = ", ".join(
            [
                field
                for field in expl_fields
                if field in learning_fields_categorical
            ]
        )
        eo[field].perform_inference(save=False)
        return eo

    def engine_object_predict(self, field, expl_fields):
        eo = self.engine_object_perform_inference(field, expl_fields.keys())
        value = eo[field].predict([expl_fields])
        return value[0]

    def perform_inference(self, save=True):
        """
        This imputer performs inference upon request for imputing a field
        from variable explanatory fields (learning fields).
        Only initialize the engine_object and set as inferred for making
        .predict() work.
        """
        self.engine_object = self.get_engine_object()
        self.is_inferred = True
        if save:
            self.save()
        return self.engine_object

    def impute_row(self, row):
        if not allNotNone(row):
            row = self.technique._observation_list_to_dict(row)
            for field in row:
                if row[field] is None:
                    non_na_fields = self.get_non_na_fields()
                    if field in non_na_fields:
                        row_non_na = {
                            field: row[field]
                            for field in row
                            if field in non_na_fields
                            and row[field] is not None
                        }
                        if not row_non_na:
                            raise ValueError(
                                _(
                                    "'row' cannot be entirely missing (either all "
                                    "None or only fields without data in the Data "
                                    "Model (na_fields)."
                                )
                            )
                        row[field] = self.engine_object_predict(
                            field, row_non_na
                        )
                    else:
                        row[field] = self.NA_COLUMN_FILL
            return self.technique._observation_dict_to_list(row)
        else:
            return row

    def get_non_na_fields(self):
        # Data non NA fields
        if not self._non_na_fields:
            self._non_na_fields = [
                f
                for f in self.get_supported_fields()
                if f not in self.technique._get_data_learning_fields_na()
            ]
        return self._non_na_fields

    def get_supported_fields(self):
        return self.technique._get_data_learning_fields_supported()

    def _get_technique_for_regression(self):
        if not self._regression_technique:
            if self.regression_technique:
                self._regression_technique = import_string(
                    self.regression_technique
                )
        return self._regression_technique

    def _get_technique_for_classification(self):
        if not self._classification_technique:
            if self.classification_technique:
                self._classification_technique = import_string(
                    self.classification_technique
                )
        return self._classification_technique
