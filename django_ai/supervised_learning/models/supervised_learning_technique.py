from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from django_ai.ai_base.models import LearningTechnique
from django_ai.ai_base.utils import allNotNone
from django_ai.ai_base.metrics import METRICS, format_metric


class SupervisedLearningTechnique(LearningTechnique):
    """
    Metaclass for Supervised Learning Techniques.
    """

    SL_TYPE_CLASSIFICATION = 0
    SL_TYPE_REGRESSION = 1

    _target_levels = None

    #: Choices for Supervised Learning Type
    SL_TYPE_CHOICES = (
        (SL_TYPE_CLASSIFICATION, "Classification"),
        (SL_TYPE_REGRESSION, "Regression"),
    )

    #: Supervised Learning Type
    sl_type = models.SmallIntegerField(
        "Supervised Learning Type",
        choices=SL_TYPE_CHOICES,
        default=SL_TYPE_CLASSIFICATION,
        blank=True,
        null=True,
    )
    # -> Cross Validation
    #: Enable Cross Validation (k-Folded)
    cv_is_enabled = models.BooleanField(
        "Cross Validation is Enabled?",
        default=False,
        help_text=("Enable Cross Validation"),
    )
    #: Quantity of Folds to be used in Cross Validation
    cv_folds = models.SmallIntegerField(
        "Cross Validation Folds",
        blank=True,
        null=True,
        help_text=("Quantity of Folds to be used in Cross Validation"),
    )
    #: Metric to be evaluated in Cross Validation (i.e. accurracy)
    cv_metrics = models.CharField(
        "Cross Validation Metric",
        max_length=255,
        blank=True,
        null=True,
        help_text=(
            "Metrics to be evaluated in Cross Validation separated by a "
            'comma and space (i.e. "accuracy, confusion_matrix, '
            'fowlkes_mallows_score")'
        ),
    )
    #: Field in the Data Model containing the targets for Learning
    learning_target = models.CharField(
        "Learning Targets",
        max_length=50,
        blank=True,
        null=True,
        help_text=(
            "Field in the Data Model containing the targets for Learning"
        ),
    )
    #: Monotonic Constraints for Learning Fields in the
    #: "field: {-1|0|1}" format separated by a comma and space, i.e.
    #: "avg1: -1, rbc: 1". Ommited fields will use 0. If left blank, the
    #: Data Model\'s one will be used. Use "None" to ensure no
    #: Monotic Constraints.
    monotonic_constraints = models.TextField(
        "Monotonic Constraints",
        blank=True,
        null=True,
        help_text=(
            "Monotonic Constraints for Learning Fields in the "
            '"field: {-1|0|1}" format separated by a comma and space, i.e. '
            '"avg1: -1, rbc: 1". Ommited fields will use 0. If left blank, the '
            'Data Model\'s one will be used. Use "None" to ensure no '
            "Monotic Constraints."
        ),
    )

    class Meta:
        verbose_name = _("Supervised Learning Technique")
        verbose_name_plural = _("Supervised Learning Techniques")
        app_label = "supervised_learning"

    def __init__(self, *args, **kwargs):
        if not args:
            kwargs["technique_type"] = self.TYPE_SUPERVISED
        super(SupervisedLearningTechnique, self).__init__(*args, **kwargs)

    def __str__(self):
        return "[SL] {0}".format(self.name)

    # -> Public API
    def engine_object_predict(self, *args, **kwargs):
        """
        Returns the prediction of the Engine's current state for the
        given data.
        """
        raise NotImplementedError("A Technique should implement this method")

    def engine_object_predict_scores(self, *args, **kwargs):
        """
        Returns the score used for prediction by the Engine's current state for the
        given data.
        """
        raise NotImplementedError("A Technique should implement this method")

    def get_targets(self):
        """
        Returns a list of labels for the data available for the model.
        """
        learning_target = self._get_data_learning_target()
        return [
            label
            for label in self._get_data_queryset().values_list(
                learning_target, flat=True
            )
        ]

    def predict(self, samples, include_scores=False):
        if self.is_inferred:
            f_samples = []
            for sample in samples:
                if isinstance(sample, dict):
                    sample = self._observation_dict_to_list(sample)
                elif isinstance(sample, (list, tuple)):
                    sample = list(sample)
                    supported_fields = (
                        self._get_data_learning_fields_supported()
                    )
                    if len(sample) < len(supported_fields):
                        # Complete it as NA values assuming the order is right
                        for x in range(len(sample), len(supported_fields)):
                            sample.append(None)
                    if not self.SUPPORTS_CATEGORICAL and self.cift_is_enabled:
                        sample = self._cift_row(sample)
                else:
                    sample = self._observation_object_to_list(sample)
                if not allNotNone(sample) and not self.SUPPORTS_NA:
                    if self.data_imputer:
                        imputer = self.get_data_imputer_object()
                        sample = imputer.impute_row(sample)
                    else:
                        sample = [
                            column for column in sample if column is not None
                        ]
                f_samples.append(sample)

            predicted = self.engine_object_predict(f_samples)
            if include_scores:
                predicted_scores = self.engine_object_predict_scores(f_samples)
                if any(predicted_scores):
                    levels = self._get_target_levels()
                    predicted_scores = [
                        ps[levels.index(predicted[index])] if ps else None
                        for index, ps in enumerate(predicted_scores)
                    ]
                return (predicted, predicted_scores)
            else:
                return predicted
        else:
            return None

    def perform_cross_validation(self):
        """
        Performs Cross Validation with the current state of the engine on the
        learning data.
        """
        cv_md = {}
        scores = self.get_cross_validation_scores()
        for metric in self._get_cv_metrics():
            cv_md[metric] = {
                "values": scores[metric].tolist(),
                "summary": format_metric(metric, scores[metric]),
            }
        return cv_md

    def get_inference_scores(self):
        scores = {}
        if self.cv_is_enabled:
            cv_scores = self.perform_cross_validation()
            scores["cv"] = cv_scores
        return scores

    def get_data_model_metadata(self):
        metadata = super().get_data_model_metadata()
        metadata["learning_target"] = self._get_data_learning_target()
        metadata[
            "monotonic_constraints"
        ] = self.monotonic_constraints or self._get_data_model_attr(
            "LEARNING_FIELDS_MONOTONIC_CONSTRAINTS", None
        )
        return metadata

    def get_inference_metadata(self):
        metadata = super().get_inference_metadata()
        metadata["conf"]["cv"] = {
            "is_enabled": self.cv_is_enabled,
            "folds": self.cv_folds,
            "metrics": self.cv_metrics,
        }
        return metadata

    def get_cross_validation_scores(self):
        """
        Returns a list with the cross validation score for each fold.
        """
        raise NotImplementedError("A Technique should implement this method")

    def engine_object_inference_scores(self):
        """
        Returns a dict with scores about the inference.
        """
        raise NotImplementedError("A Technique should implement this method")

    def _get_data_queryset(self):
        qs = super()._get_data_queryset()
        learning_target = self._get_data_learning_target()
        return qs.exclude(**{"{}__isnull".format(learning_target): True})

    def _get_data_learning_target(self):
        m_learning_target = self._get_data_model_attr("learning_target", None)
        return self.learning_target or m_learning_target

    def _get_data_monotonic_constraints(self):
        dm_monotonic_constrainsts = self._get_data_model_attr(
            "LEARNING_FIELDS_MONOTONIC_CONSTRAINTS", None
        )
        return self.monotonic_constraints or dm_monotonic_constrainsts

    def _parse_monotonic_constraints(self, csts_string):
        csts_dict = {
            field: int(val)
            for (field, val) in [
                fv.split(": ") for fv in csts_string.split(", ")
            ]
        }
        csts = []
        for field in self._get_data_learning_fields():
            csts.append(csts_dict[field] if field in csts_dict else 0)
        return csts

    def _get_monotonic_constraints_list(self):
        default_csts = [0 for f in self._get_data_learning_fields()]
        csts = self._get_data_monotonic_constraints()
        if not csts:
            return default_csts
        elif csts == "None":
            return default_csts
        else:
            return self._parse_monotonic_constraints(csts)

    def _get_target_levels(self):
        if not self._target_levels:
            levels = self._get_field_levels(self._get_data_learning_target())
            self._target_levels = levels
        return self._target_levels

    def _get_cv_metrics(self):
        """
        Returns a dict with the metrics to be evaluated post-inference. Should be
        overriden to customize the metrics.
        """
        if self.cv_metrics:
            return self.cv_metrics.split(", ")
        else:
            return []

    def _get_metadata_descriptions(self):
        descriptions = super()._get_metadata_descriptions()
        descriptions["cv"] = "Cross Validation"
        descriptions["is_enabled"] = "Is Enabled"
        descriptions["folds"] = "Folds"
        descriptions["metrics"] = "Metrics"
        descriptions["monotonic_constraints"] = "Monotonic Constraints"
        descriptions["learning_target"] = "Learning Target"
        # Generated by format_metric
        descriptions["summary"] = "Summary"
        descriptions["values"] = "Values"
        return descriptions

    # -> Django Models API
    def clean(self):
        super().clean()
        # -> Check validity of learning labels
        data_model = self._get_data_model()
        if self.learning_target:
            try:
                getattr(data_model, self.learning_target)
            except Exception:
                raise ValidationError(
                    {
                        "learning_target": _(
                            "Unrecognized field in model {} for Learning Labels: {}".format(
                                self.data_model, self.learning_target
                            )
                        )
                    }
                )
        else:
            learning_target = self._get_data_model_attr(
                "learning_target", None
            )
            if not learning_target:
                raise ValidationError(
                    {
                        "data_model": _(
                            "The model must provide learning_target if no Learning "
                            "labels are defined in the technique"
                        )
                    }
                )
            try:
                getattr(data_model, learning_target)
            except Exception:
                raise ValidationError(
                    {
                        "data_model": _(
                            "Unrecognized field in Learning Labels: {}".format(
                                learning_target
                            )
                        )
                    }
                )
        if not self.monotonic_constraints:
            monotonic_constraints = self._get_data_model_attr(
                "LEARNING_FIELDS_MONOTONIC_CONSTRAINTS", None
            )
            if monotonic_constraints:
                if monotonic_constraints != "None":
                    key_vals = monotonic_constraints.split(", ")
                    for key_val in key_vals:
                        key, val = key_val.split(": ")
                        if key not in self._get_data_learning_fields():
                            raise ValidationError(
                                {
                                    "data_model": _(
                                        "Unrecognized field in Monotonic Constraints: "
                                        "{}".format(key)
                                    )
                                }
                            )
                        if val not in ["-1", "0", "1"]:
                            raise ValidationError(
                                {
                                    "data_model": _(
                                        "Unrecognized value in Monotonic Constraints: "
                                        "{}".format(val)
                                    )
                                }
                            )
        elif self.monotonic_constraints != "None":
            key_vals = self.monotonic_constraints.split(", ")
            for key_val in key_vals:
                try:
                    key, val = key_val.split(": ")
                    if key not in self._get_data_learning_fields():
                        raise ValidationError(
                            {
                                "monotonic_constraints": _(
                                    "Unrecognized field in Monotonic Constraints: {}".format(
                                        key
                                    )
                                )
                            }
                        )
                    if val not in ["-1", "0", "1"]:
                        raise ValidationError(
                            {
                                "monotonic_constraints": _(
                                    "Unrecognized value in Monotonic Constraints: {}".format(
                                        val
                                    )
                                )
                            }
                        )
                except ValueError:
                    raise ValidationError(
                        {
                            "monotonic_constraints": _(
                                "Unrecognized format in Monotonic Constraints:"
                            )
                        }
                    )
        if self.cv_is_enabled:
            if not self.cv_folds:
                raise ValidationError(
                    {
                        "cv_folds": _(
                            "The field is required if Cross Validation is enabled"
                        )
                    }
                )
            if not self.cv_metrics:
                raise ValidationError(
                    {
                        "cv_metrics": _(
                            "The field is required if Cross Validation is enabled"
                        )
                    }
                )
            else:
                for metric in self.cv_metrics.split(", "):
                    if metric not in METRICS.keys():
                        raise ValidationError(
                            {
                                "cv_metrics": _(
                                    "Unrecognized metric: {}".format(metric)
                                )
                            }
                        )
