# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (cross_val_score, )

from django.db import models
from django.utils import timezone
from django.utils.translation import ugettext_lazy as _


if 'DJANGO_TEST' in os.environ:
    from django_ai.ai_base.models import SupervisedLearningTechnique
else:  # pragma: no cover
    from django_ai.ai_base.models import SupervisedLearningTechnique


class HGBTree(SupervisedLearningTechnique):
    """
    Histogram-based Gradient Boosting Tree - Classification

    """
    LOSS_CHOICES = (
        ('auto', "Automatic"),
        ('binary_crossentropy', "Binary Cross-Entropy"),
        ('categorical_crossentropy', "Categorical Cross-Entropy"),
    )

    EARLY_STOPPING_CHOICES = (
        ('auto', "Automatic"),
        ('true', "True"),
        ('false', "False"),
    )

    CV_CHOICES = (
        ('accuracy', _("Accuracy")),
        ('average_precision', _("Average Precision")),
        ('f1', _("F1")),
        ('neg_log_loss', _("Logistic Loss")),
        ('precision', _("Precision")),
        ('recall', _("Recall")),
        ('roc_auc', _("Area under ROC Curve")),
    )

    loss = models.CharField(
        "Loss function",
        choices=LOSS_CHOICES, default='auto',
        blank=True, null=True, max_length=50,
        help_text=(
            'The loss function to be used in the boosting process. '
            'If none is given, auto will be used.'
        )
    )

    learning_rate = models.FloatField(
        "Learning Rate",
        default=0.1, blank=True, null=True,
        help_text=(
            'The leraning rate, a.k.a. shrinkage. Multiplicative factor '
            'for the leaves values. Use 1 for no shrinkage.'
        )
    )

    max_leaf_nodes = models.IntegerField(
        "Maximum number of leaves for each tree",
        default=31, blank=True, null=True,
        help_text=(
            'If not None, must be strictly greater than 1.'
        )
    )

    max_depth = models.IntegerField(
        "Maximum depth of each tree",
        default=None, blank=True, null=True,
        help_text=(
            'Depth is not constrained by default.'
        )
    )

    min_samples_leaf = models.IntegerField(
        "Minimum number of samples per leaf",
        default=20, blank=True, null=True,
        help_text=(
            'For datasets with less than hundred of samples, it is '
            'recommended to be lower than 20.'
        )
    )

    l2_regularization = models.FloatField(
        "L2 Regularization Parameter",
        default=0, blank=True, null=True,
        help_text=(
            'Use 0 for no regularization.'
        )
    )

    max_bins = models.IntegerField(
        "Maximum number of bins for non-missing values",
        default=255, blank=True, null=True,
        help_text=(
            'Must be no larger than 255.'
        )
    )

    warm_start = models.BooleanField(
        "Warm Start",
        default=False,
        help_text=(
            'When set to True, reuse the solution of the previous call '
            'to fit and add more estimators to the ensemble. For results '
            'to be valid, the estimator should be re-trained on the same '
            'data only.'
        )
    )

    early_stopping = models.CharField(
        "Early Stopping (ES)",
        choices=EARLY_STOPPING_CHOICES, default='auto',
        blank=True, null=True, max_length=50,
        help_text=(
            'If auto, early stopping is enabled if the sample size is greater '
            'than 10000. If true, is enabled, otherwise disabled.'
        )
    )

    scoring = models.CharField(
        "Scoring Paramenter for ES",
        default='loss',
        blank=True, null=True, max_length=50,
        help_text=(
            'If None, the estimator\'s default scorer is used. Only used if '
            'early stopping is performed.'
        )
    )

    validation_fraction = models.FloatField(
        "Validation Fraction for ES",
        default=0.1, blank=True, null=True,
        help_text=(
            'Proportion of training data for validating early stopping. '
            'Only used if early stopping is performed.'
        )
    )

    n_iter_no_change = models.IntegerField(
        "Iterations without Change for ES",
        default=10, blank=True, null=True,
        help_text=(
            'Used to determine when to "Early Stop". '
            'Only used if early stopping is performed.'
        )
    )

    tol = models.FloatField(
        "Abs. Tol. for comparing Scores for ES",
        default="1e-7", blank=True, null=True,
        help_text=(
            'The higher the tolerance, the more likely to "early stop". '
            'Only used if early stopping is performed.'
        )
    )

    verbose = models.IntegerField(
        "Verbosity Level",
        default=0, blank=True, null=True,
        help_text=(
            'If not zero, print some information about the fitting process. '
            '(currently STDOUT in the Django process)'
        )
    )

    random_state = models.IntegerField(
        "Random State seed number",
        default=None, blank=True, null=True,
        help_text=(
            'Use a number for reproducible output across multiple function '
            'calls'
        )
    )

    # -> Cross Validation
    #: Metric to be evaluated in Cross Validation
    cv_metric = models.CharField(
        "Cross Validation Metric",
        max_length=20, blank=True, null=True, choices=CV_CHOICES,
        help_text=(
            'Metric to be evaluated in Cross Validation'
        )
    )

    # #: Auto-generated Image if available
    # image = models.ImageField(
    #     "Image",
    #     blank=True, null=True,
    #     help_text=(
    #         'Auto-generated Image if available'
    #     )
    # )

    class Meta:
        verbose_name = "HGB Tree for Classification"
        verbose_name_plural = "HGB Trees for Classification"
        app_label = "supervised_learning"

    def save(self, *args, **kwargs):
        # Initialize metadata field if corresponds
        if self.metadata == {}:
            self.metadata["current_inference"] = {}
            self.metadata["previous_inference"] = {}
        super().save(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        kwargs["sl_type"] = self.SL_TYPE_CLASSIFICATION
        super(HGBTree, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[HGBTree|C] {0}".format(self.name))

    def get_engine_object(self, reconstruct=False, save=True):
        if self.engine_object is not None and not reconstruct:
            return(self.engine_object)

        # -> Ensure defaults
        max_iters = self.engine_iterations
        if not max_iters:
            max_iters = 100
        categorical_features = self.get_categorical_mask()
        monotonic_csts = self.get_monotonic_constraints()
        classifier = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=max_iters,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_csts,
            warm_start=self.warm_start,
            early_stopping=self.early_stopping,
            scoring=self.scoring,  # self.verbose,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=float(self.tol),
            verbose=self.verbose,
            random_state=self.random_state
        )
        self.engine_object = classifier
        if save:
            self.save()
        return(self.engine_object)

    def perform_inference(self, recalculate=False, save=True):
        if not self.is_inferred or recalculate:
            # No need for running the inference 'engine_meta_iterations' times
            eo = self.get_engine_object(reconstruct=True)
            # -> Get the data
            data = self.get_data()
            # -> Get the labels
            labels = self.get_labels()
            # -> Run the algorithm and store the updated engine object
            self.engine_object = eo.fit(data, labels)
            # -> Rotate metadata
            self.rotate_metadata()
            # -> Perform Cross Validation
            if self.cv_is_enabled:
                self.perform_cross_validation(data=data, labels=labels,
                                              update_metadata=True)
            # -> Update other metadata
            self.metadata["current_inference"]["input_dimensionality"] = \
                np.shape(data)
            self.metadata["current_inference"]["classifier_conf"] = \
                self.get_conf_dict()
            self.metadata["current_inference"]["score"] = \
                float(self.engine_object.score(data, labels))
            # -> Set as inferred
            self.is_inferred = True
            if save:
                self.engine_object_timestamp = timezone.now()
                self.save()
        return(self.engine_object)

    def get_conf_dict(self):
        conf_dict = {}
        conf_dict['name'] = self.name
        conf_dict['params'] = self.get_engine_object().get_params()
        return(conf_dict)

    def get_labels(self):
        return list(super().get_labels())

    def get_categorical_mask(self):
        return list(self.data_columns.all().order_by("position").values_list(
            "is_categorical", flat=True))

    def get_monotonic_constraints(self):
        return list(self.data_columns.all().order_by("position").values_list(
            "monotonic_cst", flat=True))

    def perform_cross_validation(self, data=None, labels=None,
                                 update_metadata=False):
        if data is None:
            data = self.get_data()
        if labels is None:
            labels = self.get_labels()
        classifier = self.get_engine_object()
        scores = cross_val_score(
            classifier, data, labels,
            cv=self.cv_folds, scoring=self.cv_metric
        )
        if update_metadata:
            self.metadata["current_inference"]['cv'] = {}
            self.metadata["current_inference"]['cv']['conf'] = {
                "folds": self.cv_folds,
                "metric": self.get_cv_metric_display()
            }
            self.metadata["current_inference"]['cv']['scores'] = list(scores)
            self.metadata["current_inference"]['cv']['mean'] = scores.mean()
            self.metadata["current_inference"]['cv']['2std'] = 2 * scores.std()
        return(scores)
