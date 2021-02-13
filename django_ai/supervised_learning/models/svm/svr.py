from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from django.db import models
from django.utils.translation import ugettext_lazy as _

from django_ai.supervised_learning.models import SupervisedLearningTechnique


class SVR(SupervisedLearningTechnique):
    """
    Support Vector Machine - Regression

    """
    SUPPORTS_NA = False
    SUPPORTS_CATEGORICAL = False
    PREDICT_SCORE_TYPE = None
    LEARNING_PAIR = 'django_ai.supervised_learning.models.SVC'

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
        verbose_name = _("Supzport Vector Machine - Regression")
        verbose_name_plural = _("Support Vector Machines - Regression")
        app_label = "supervised_learning"

    def __init__(self, *args, **kwargs):
        kwargs["sl_type"] = self.SL_TYPE_REGRESSION
        super(SVR, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[SVM|R] {0}".format(self.name))

    def engine_object_init(self):
        # -> Ensure defaults
        gamma = self.kernel_coefficient
        max_iters = self.max_iter
        if not max_iters:
            max_iters = -1
        #
        regressor = svm.SVR(
            C=self.penalty_parameter,
            kernel=self.kernel,
            degree=self.kernel_poly_degree,
            gamma=gamma,
            coef0=self.kernel_independent_term,
            shrinking=self.use_shrinking,
            tol=float(self.tolerance),
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=max_iters,
        )
        pipe = make_pipeline(StandardScaler(), regressor)
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
        return [None for sample in samples]

    def get_engine_object_conf(self):
        conf = self.get_engine_object().get_params()
        conf.pop('steps')
        conf.pop('standardscaler')
        conf.pop('svr')
        return conf

    def get_inference_scores(self):
        scores = super().get_inference_scores()
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
