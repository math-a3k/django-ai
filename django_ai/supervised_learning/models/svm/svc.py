from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from django.db import models
from django.utils.translation import gettext_lazy as _

from django_ai.supervised_learning.models import SupervisedLearningTechnique


class SVC(SupervisedLearningTechnique):
    """
    Support Vector Machine - Classification

    """
    SUPPORTS_NA = False
    SUPPORTS_CATEGORICAL = False
    PREDICT_SCORE_TYPE = _("Probability")
    LEARNING_PAIR = 'django_ai.supervised_learning.models.SVR'

    SVM_KERNEL_CHOICES = (
        ('linear', _("Linear")),
        ('poly', _("Polynomial")),
        ('rbf', _("RBF")),
        ('linear', _("Linear")),
        ('sigmoid', _("Sigmoid")),
        ('precomputed', _("Precomputed")),
    )

    # (skl) C : float, optional (default=1.0)
    #: Penalty parameter (C) of the error term.
    penalty_parameter = models.FloatField(
        _("Penalty Parameter"),
        default=1.0, blank=True, null=True,
        help_text=(_(
            'Penalty parameter (C) of the error term.'
        ))
    )

    # (skl) kernel : string, optional (default=’rbf’)
    #: Kernel to be used in the SVM. If none is given, RBF will be used.
    kernel = models.CharField(
        _("SVM Kernel"),
        choices=SVM_KERNEL_CHOICES, default='rbf',
        blank=True, null=True, max_length=50,
        help_text=(_(
            'Kernel to be used in the SVM. If none is given, RBF will be '
            'used.'
        ))
    )

    # (skl) degree : int, optional (default=3)
    #: Degree of the Polynomial Kernel function. Ignored
    #: by all other kernels.
    kernel_poly_degree = models.IntegerField(
        _("Polynomial Kernel degree"),
        default=3, blank=True, null=True,
        help_text=(_(
            'Degree of the Polynomial Kernel function. Ignored '
            'by all other Kernels.'
        ))
    )

    # (skl) gamma : 'scale', 'auto' or float, optional (default=’scale’)
    #: Kernel coefficient for RBF, Polynomial and Sigmoid.
    #: Leave blank "for automatic" (1/n_features will be used)
    kernel_coefficient = models.CharField(
        _("Kernel coefficient"),
        blank=True, null=True, max_length=20,
        default="scale",
        help_text=(_(
            'Kernel coefficient for RBF, Polynomial and Sigmoid. '
            '"scale", "auto" or a float are supported.'
        ))
    )

    # (skl) coef0 : float, optional (default=0.0)
    #: Independent term in kernel function. It is only significant
    #: in Polynomial and Sigmoid kernels.
    kernel_independent_term = models.FloatField(
        _("Kernel Independent Term"),
        default=0.0, blank=True, null=True,
        help_text=(_(
            'Independent term in kernel function. It is only significant '
            'in Polynomial and Sigmoid kernels.'
        ))
    )

    # (skl) probability : boolean, optional (default=False)
    #: Whether to enable probability estimates. This will slow
    #: model fitting.
    estimate_probability = models.BooleanField(
        _("Estimate Probability?"),
        default=True,
        help_text=(_(
            'Whether to enable probability estimates. This will slow '
            'model fitting.'
        ))
    )

    # (skl) shrinking : boolean, optional (default=True)
    #: Whether to use the shrinking heuristic.
    use_shrinking = models.BooleanField(
        _("Use Shrinking Heuristic?"),
        default=True,
        help_text=(_(
            'Whether to use the shrinking heuristic.'
        ))
    )

    # (skl) tol : float, optional (default=1e-3)
    #: Tolerance for stopping criterion.
    tolerance = models.FloatField(
        _("Tolerance"),
        default="1e-3", blank=True, null=True,
        help_text=(_(
            "Tolerance for stopping criterion."
        ))
    )

    # cache_size : float, optional
    #: Specify the size of the kernel cache (in MB).
    cache_size = models.FloatField(
        _('Kernel Cache Size (MB)'),
        blank=True, null=True,
        default=200,
        help_text=(_(
            'Specify the size of the kernel cache (in MB).'
        ))
    )

    # (skl) class_weight : {dict, ‘balanced’}, optional
    #: Set the parameter C of class i to class_weight[i]*C for SVC.
    #: If not given, all classes are supposed to have weight one. The
    #: “balanced” mode uses the values of y to automatically adjust
    #: weights inversely proportional to class frequencies in the
    #: input data as n_samples / (n_classes * np.bincount(y))
    class_weight = models.CharField(
        _('Class Weight'),
        max_length=50, blank=True, null=True,
        help_text=(_(
            'Set the parameter C of class i to class_weight[i]*C for SVC. '
            'If not given, all classes are supposed to have weight one. The '
            '“balanced” mode uses the values of y to automatically adjust '
            'weights inversely proportional to class frequencies in the '
            'input data as n_samples / (n_classes * np.bincount(y))'
        ))
    )

    # (skl) verbose : bool, default: False
    #: Enable verbose output. Note that this setting takes advantage
    #: of a per-process runtime setting in libsvm that, if enabled,
    #: may not work properly in a multithreaded context.
    verbose = models.BooleanField(
        _('Be Verbose?'),
        default=False,
        help_text=(_(
            'Enable verbose output. Note that this setting takes advantage '
            'of a per-process runtime setting in libsvm that, if enabled, '
            'may not work properly in a multithreaded context.'
        ))
    )

    # (skl) decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
    #: Whether to return a one-vs-rest (‘ovr’) decision function of
    #: shape (n_samples, n_classes) as all other classifiers, or the
    #: original one-vs-one (‘ovo’) decision function of libsvm which
    #: has shape (n_samples, n_classes * (n_classes - 1) / 2).
    decision_function_shape = models.CharField(
        _('Decision Function Shape'),
        choices=(('ovo', 'One-VS-One'), ('ovr', 'One-VS-Rest')),
        default='ovr', max_length=10, blank=True, null=True,
        help_text=(_(
            'Whether to return a one-vs-rest (‘ovr’) decision function of '
            'shape (n_samples, n_classes) as all other classifiers, or the '
            'original one-vs-one (‘ovo’) decision function of libsvm which '
            'has shape (n_samples, n_classes * (n_classes - 1) / 2).'
        ))
    )

    # (skl) random_state : int, RandomState instance or None,
    #                      optional (default=None)
    #: The seed of the pseudo random number generator to use when
    #: shuffling the data. If int, random_state is the seed used by the
    #: random number generator; If RandomState instance, random_state
    #: is the random number generator; If None, the random number
    #: generator is the RandomState instance used by np.random.
    random_state = models.IntegerField(
        _("Random Seed"),
        blank=True, null=True,
        help_text=(_(
            'The seed of the pseudo random number generator to use when '
            'shuffling the data. If int, random_state is the seed used by the '
            'random number generator; If RandomState instance, random_state '
            'is the random number generator; If None, the random number '
            'generator is the RandomState instance used by np.random.'
        ))
    )

    # (skl) max_iter : int, (default=10000)
    max_iter = models.IntegerField(
        _("Maximum Iterations Safeguard"),
        default=10000,
        blank=True, null=True,
        help_text=(_(
            'Stop if Maximum Iterations has been reached without '
            'meeting algorithm converge conditions'
        ))
    )

    #: Auto-generated Images if available
    images = models.ImageField(
        _("Image"),
        blank=True, null=True,
        help_text=(_(
            'Auto-generated Image if available'
        ))
    )

    class Meta:
        verbose_name = _("Support Vector Machine - Classification")
        verbose_name_plural = _("Support Vector Machines - Classification")
        app_label = "supervised_learning"

    def __init__(self, *args, **kwargs):
        kwargs["sl_type"] = self.SL_TYPE_CLASSIFICATION
        super(SVC, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[SVM|C] {0}".format(self.name))

    def engine_object_init(self):
        # -> Ensure defaults
        gamma = self.kernel_coefficient
        max_iters = self.max_iter
        if not max_iters:
            max_iters = -1
        #
        classifier = svm.SVC(
            C=self.penalty_parameter,
            kernel=self.kernel,
            degree=self.kernel_poly_degree,
            gamma=gamma,
            coef0=self.kernel_independent_term,
            shrinking=self.use_shrinking,
            probability=self.estimate_probability,
            tol=float(self.tolerance),
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            verbose=self.verbose,
            max_iter=max_iters,
            decision_function_shape=self.decision_function_shape,
            random_state=self.random_state
        )
        pipe = make_pipeline(StandardScaler(), classifier)
        return pipe

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
        if self.estimate_probability:
            eo = self.get_engine_object()
            return eo.predict_proba(samples).tolist()
        else:
            return [None for sample in samples]

    def get_engine_object_conf(self):
        conf = self.get_engine_object().get_params()
        conf.pop('steps')
        conf.pop('standardscaler')
        conf.pop('svc')
        return conf

    def get_inference_scores(self):
        scores = super().get_inference_scores()
        data = self.get_data()
        labels = self.get_targets()
        eo = self.get_engine_object()
        scores["ldra"] = float(eo.score(data, labels))
        return scores

    def get_cross_validation_scores(self):
        metrics = self._get_cv_metrics()
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
