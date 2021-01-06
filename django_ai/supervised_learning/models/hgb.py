# -*- coding: utf-8 -*-

import os

from django.db import models

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

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

    def __init__(self, *args, **kwargs):
        kwargs["sl_type"] = self.SL_TYPE_CLASSIFICATION
        super(HGBTree, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[HGBTree|C] {0}".format(self.name))

    def get_engine_object(self):
        # -> Ensure defaults
        # gamma = self.kernel_coefficient
        # if not gamma:
        #     gamma = 'auto'
        max_iters = self.engine_iterations
        if not max_iters:
            max_iters = 100
        # cache_size = self.cache_size
        # if not cache_size:
        #     cache_size = 200
        classifier = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.self.learning_rate,
            max_iter=max_iters,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            categorical_features=None,  # <-- TODO
            monotonic_cst=None,  # <-- TODO
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
        return(classifier)

    def get_conf_dict(self):
        conf_dict = {}
        conf_dict['name'] = self.name
        # conf_dict['kernel'] = self.get_kernel_display()
        # kernel_details = ""
        # if self.kernel == "poly":
        #     kernel_details += "Degree: {} ".format(
        #         self.kernel_poly_degree)
        # if (self.kernel == "poly" or self.kernel == "sigmoid" or
        #         self.kernel == "rbf"):
        #     kc = self.kernel_coefficient if self.kernel_coefficient else "Auto"
        #     kernel_details += "Coef. (gamma): {} - ".format(kc)
        # if (self.kernel == "poly" or self.kernel == "sigmoid"):
        #     kit = self.kernel_independent_term \
        #         if self.kernel_independent_term else "0"
        #     kernel_details += "Indep. Term: {} - ".format(kit)
        # conf_dict['kernel_details'] = kernel_details
        # conf_dict['penalty_parameter'] = self.penalty_parameter
        # conf_dict['str'] = "Kernel: {}{}, Penalty: {}".format(
        #     self.get_kernel_display(), kernel_details, self.penalty_parameter
        # )
        return(conf_dict)
