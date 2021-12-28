from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

from django.db import models
from django.utils.translation import gettext_lazy as _

from django_ai.ai_base.metrics import METRICS
from django_ai.supervised_learning.models.supervised_learning_technique \
    import SupervisedLearningTechnique


class HGBTreeClassifier(SupervisedLearningTechnique):
    """
    Histogram-based Gradient Boosting Tree - Classification

    """
    SUPPORTS_NA = True
    SUPPORTS_CATEGORICAL = True
    PREDICT_SCORE_TYPE = _("Probability")
    LEARNING_PAIR = 'django_ai.supervised_learning.models.HGBTreeRegressor'

    LOSS_CHOICES = (
        ('auto', _("Automatic")),
        ('binary_crossentropy', _("Binary Cross-Entropy")),
        ('categorical_crossentropy', _("Categorical Cross-Entropy")),
    )

    EARLY_STOPPING_CHOICES = (
        ('auto', _("Automatic")),
        ('true', _("True")),
        ('false', _("False")),
    )

    loss = models.CharField(
        _("Loss function"),
        choices=LOSS_CHOICES, default='auto',
        blank=True, null=True, max_length=50,
        help_text=(_(
            'The loss function to be used in the boosting process. '
            'If none is given, auto will be used.'
        ))
    )

    learning_rate = models.FloatField(
        _("Learning Rate"),
        default=0.1, blank=True, null=True,
        help_text=(_(
            'The leraning rate, a.k.a. shrinkage. Multiplicative factor '
            'for the leaves values. Use 1 for no shrinkage.'
        ))
    )

    max_leaf_nodes = models.IntegerField(
        _("Maximum number of leaves for each tree"),
        default=31, blank=True, null=True,
        help_text=(_(
            'If not None, must be strictly greater than 1.'
        ))
    )

    max_depth = models.IntegerField(
        _("Maximum depth of each tree"),
        default=None, blank=True, null=True,
        help_text=(_(
            'Depth is not constrained by default.'
        ))
    )

    min_samples_leaf = models.IntegerField(
        _("Minimum number of samples per leaf"),
        default=20, blank=True, null=True,
        help_text=(_(
            'For datasets with less than hundred of samples, it is '
            'recommended to be lower than 20.'
        ))
    )

    l2_regularization = models.FloatField(
        _("L2 Regularization Parameter"),
        default=0, blank=True, null=True,
        help_text=(_(
            'Use 0 for no regularization.'
        ))
    )

    max_bins = models.IntegerField(
        _("Maximum number of bins for non-missing values"),
        default=255, blank=True, null=True,
        help_text=(_(
            'Must be no larger than 255.'
        ))
    )

    warm_start = models.BooleanField(
        _("Warm Start"),
        default=False,
        help_text=(_(
            'When set to True, reuse the solution of the previous call '
            'to fit and add more estimators to the ensemble. For results '
            'to be valid, the estimator should be re-trained on the same '
            'data only.'
        ))
    )

    early_stopping = models.CharField(
        _("Early Stopping (ES)"),
        choices=EARLY_STOPPING_CHOICES, default='auto',
        blank=True, null=True, max_length=50,
        help_text=(_(
            'If auto, early stopping is enabled if the sample size is greater '
            'than 10000. If true, is enabled, otherwise disabled.'
        ))
    )

    scoring = models.CharField(
        _("Scoring Paramenter for ES"),
        default='loss',
        blank=True, null=True, max_length=50,
        help_text=(_(
            'If None, the estimator\'s default scorer is used. Only used if '
            'early stopping is performed.'
        ))
    )

    validation_fraction = models.FloatField(
        _("Validation Fraction for ES"),
        default=0.1, blank=True, null=True,
        help_text=(_(
            'Proportion of training data for validating early stopping. '
            'Only used if early stopping is performed.'
        ))
    )

    n_iter_no_change = models.IntegerField(
        _("Iterations without Change for ES"),
        default=10, blank=True, null=True,
        help_text=(_(
            'Used to determine when to "Early Stop". '
            'Only used if early stopping is performed.'
        ))
    )

    tol = models.FloatField(
        _("Abs. Tol. for comparing Scores for ES"),
        default="1e-7", blank=True, null=True,
        help_text=(_(
            'The higher the tolerance, the more likely to "early stop". '
            'Only used if early stopping is performed.'
        ))
    )

    verbose = models.IntegerField(
        _("Verbosity Level"),
        default=0, blank=True, null=True,
        help_text=(_(
            'If not zero, print some information about the fitting process. '
            '(currently STDOUT in the Django process)'
        ))
    )

    random_state = models.IntegerField(
        _("Random State seed number"),
        default=None, blank=True, null=True,
        help_text=(_(
            'Use a number for reproducible output across multiple function '
            'calls'
        ))
    )

    max_iter = models.IntegerField(
        _("Maximum number of iterations"),
        default=100, blank=True, null=True,
        help_text=(_(
            'Maximum number of iterations'
        ))
    )

    class Meta:
        verbose_name = _("HGB Tree for Classification")
        verbose_name_plural = _("HGB Trees for Classification")
        app_label = "supervised_learning"

    def __init__(self, *args, **kwargs):
        kwargs["sl_type"] = self.SL_TYPE_CLASSIFICATION
        super(HGBTreeClassifier, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[HGBTree|C] {0}".format(self.name))

    def engine_object_init(self):
        # -> Ensure defaults
        categorical_features = self._get_categorical_mask()
        params = {
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'max_leaf_nodes': self.max_leaf_nodes,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'l2_regularization': self.l2_regularization,
            'max_bins': self.max_bins,
            'categorical_features': categorical_features,
            'warm_start': self.warm_start,
            'early_stopping': self.early_stopping,
            'scoring': self.scoring,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'tol': float(self.tol),
            'verbose': self.verbose,
            'random_state': self.random_state
        }
        target_levels = self._get_target_levels()
        if len(target_levels) == 2:
            monotonic_csts = self._get_monotonic_constraints_list()
            params['monotonic_cst'] = monotonic_csts
        classifier = HistGradientBoostingClassifier(**params)
        return classifier

    def engine_object_perform_inference(self):
        data = self.get_data()
        labels = self.get_targets()
        eo = self.get_engine_object(reconstruct=True)
        eo.fit(data, labels)
        return eo

    def engine_object_predict(self, samples):
        eo = self.get_engine_object()
        return eo.predict(samples)

    def engine_object_predict_scores(self, samples):
        eo = self.get_engine_object()
        return eo.predict_proba(samples).tolist()

    def get_engine_object_conf(self):
        eo = self.get_engine_object()
        return eo.get_params()

    def get_inference_scores(self):
        scores = super().get_inference_scores()
        data = self.get_data()
        labels = self.get_targets()
        eo = self.get_engine_object()
        scores["ldra"] = float(eo.score(data, labels))
        return scores

    def get_cross_validation_scores(self):
        metrics = {key: METRICS[key] for key in self._get_cv_metrics()}
        data = self.get_data()
        labels = self.get_targets()
        classifier = self.get_engine_object()
        raw_scores = cross_validate(
            classifier, data, labels,
            cv=self.cv_folds, scoring=metrics
        )
        scores = {}
        for metric in metrics:
            scores[metric] = raw_scores.get("test_{}".format(metric))
        return scores

    def _get_metadata_descriptions(self):
        descriptions = super()._get_metadata_descriptions()
        descriptions["ldra"] = "Learning Data Re-classification Accuracy"
        return descriptions
