# -*- coding: utf-8 -*-

import os

from django.db import models

from sklearn import svm

if 'DJANGO_TEST' in os.environ:
    from django_ai.base.models import SupervisedLearningTechnique
else:  # pragma: no cover
    from base.models import SupervisedLearningTechnique


class SVC(SupervisedLearningTechnique):
    """
    Support Vector Machine - Classification

    """
    SVM_KERNEL_CHOICES = (
        ('linear', "Linear"),
        ('poly', "Polynomial"),
        ('rbf', "RBF"),
        ('linear', "Linear"),
        ('sigmoid', "Sigmoid"),
        ('precomputed', "Precomputed"),
    )

    # (skl) C : float, optional (default=1.0)
    #: Penalty parameter (C) of the error term.
    penalty_parameter = models.FloatField(
        "Penalty Parameter",
        default=1.0, blank=True, null=True,
        help_text=(
            'Penalty parameter (C) of the error term.'
        )
    )

    # (skl) kernel : string, optional (default=’rbf’)
    #: Kernel to be used in the SVM. If none is given, RBF will be used.
    kernel = models.CharField(
        "SVM Kernel",
        choices=SVM_KERNEL_CHOICES, default='rbf',
        blank=True, null=True, max_length=50,
        help_text=(
            'Kernel to be used in the SVM. If none is given, RBF will be used.'
        )
    )

    # (skl) degree : int, optional (default=3)
    #: Degree of the Polynomial Kernel function. Ignored
    #: by all other kernels.
    kernel_poly_degree = models.IntegerField(
        "Polynomial Kernel degree",
        default=3, blank=True, null=True,
        help_text=(
            'Degree of the Polynomial Kernel function. Ignored '
            'by all other Kernels.'
        )
    )

    # (skl) gamma : float, optional (default=’auto’)
    #: Kernel coefficient for RBF, Polynomial and Sigmoid.
    #: Leave blank "for automatic" (1/n_features will be used)
    kernel_coefficient = models.FloatField(
        "Kernel coefficient",
        blank=True, null=True,
        help_text=(
            'Kernel coefficient for RBF, Polynomial and Sigmoid. '
            'Leave blank "for automatic" (1/n_features will be used)'
        )
    )

    # (skl) coef0 : float, optional (default=0.0)
    #: Independent term in kernel function. It is only significant
    #: in Polynomial and Sigmoid kernels.
    kernel_independent_term = models.FloatField(
        "Kernel Independent Term",
        default=0.0, blank=True, null=True,
        help_text=(
            'Independent term in kernel function. It is only significant '
            'in Polynomial and Sigmoid kernels.'
        )
    )

    # (skl) probability : boolean, optional (default=False)
    #: Whether to enable probability estimates. This will slow
    #: model fitting.
    estimate_probability = models.BooleanField(
        "Estimate Probability?",
        default=False,
        help_text=(
            'Whether to enable probability estimates. This will slow '
            'model fitting.'
        )
    )

    # (skl) shrinking : boolean, optional (default=True)
    #: Whether to use the shrinking heuristic.
    use_shrinking = models.BooleanField(
        "Use Shrinking Heuristic?",
        default=True,
        help_text=(
            'Whether to use the shrinking heuristic.'
        )
    )

    # (skl) tol : float, optional (default=1e-3)
    #: Tolerance for stopping criterion.
    tolerance = models.FloatField(
        "Tolerance",
        default="1e-3", blank=True, null=True,
        help_text=(
            "Tolerance for stopping criterion."
        )
    )

    # cache_size : float, optional
    #: Specify the size of the kernel cache (in MB).
    cache_size = models.FloatField(
        'Kernel Cache Size (MB)',
        blank=True, null=True,
        help_text=('Specify the size of the kernel cache (in MB).'),
    )

    # (skl) class_weight : {dict, ‘balanced’}, optional
    #: Set the parameter C of class i to class_weight[i]*C for SVC.
    #: If not given, all classes are supposed to have weight one. The
    #: “balanced” mode uses the values of y to automatically adjust
    #: weights inversely proportional to class frequencies in the
    #: input data as n_samples / (n_classes * np.bincount(y))
    class_weight = models.CharField(
        'Class Weight',
        max_length=50, blank=True, null=True,
        help_text=(
            'Set the parameter C of class i to class_weight[i]*C for SVC. '
            'If not given, all classes are supposed to have weight one. The '
            '“balanced” mode uses the values of y to automatically adjust '
            'weights inversely proportional to class frequencies in the '
            'input data as n_samples / (n_classes * np.bincount(y))'
        ),
    )

    # (skl) verbose : bool, default: False
    #: Enable verbose output. Note that this setting takes advantage
    #: of a per-process runtime setting in libsvm that, if enabled,
    #: may not work properly in a multithreaded context.
    verbose = models.BooleanField(
        'Be Verbose?',
        default=False,
        help_text=(
            'Enable verbose output. Note that this setting takes advantage '
            'of a per-process runtime setting in libsvm that, if enabled, '
            'may not work properly in a multithreaded context.'
        ),
    )

    # (skl) decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
    #: Whether to return a one-vs-rest (‘ovr’) decision function of
    #: shape (n_samples, n_classes) as all other classifiers, or the
    #: original one-vs-one (‘ovo’) decision function of libsvm which
    #: has shape (n_samples, n_classes * (n_classes - 1) / 2).
    decision_function_shape = models.CharField(
        'Decision Function Shape',
        choices=(('ovo', 'One-VS-One'), ('ovr', 'One-VS-Rest')),
        default='ovr', max_length=10, blank=True, null=True,
        help_text=(
            'Whether to return a one-vs-rest (‘ovr’) decision function of '
            'shape (n_samples, n_classes) as all other classifiers, or the '
            'original one-vs-one (‘ovo’) decision function of libsvm which '
            'has shape (n_samples, n_classes * (n_classes - 1) / 2).'
        ),
    )

    # (skl) random_state : int, RandomState instance or None,
    #                      optional (default=None)
    #: The seed of the pseudo random number generator to use when
    #: shuffling the data. If int, random_state is the seed used by the
    #: random number generator; If RandomState instance, random_state
    #: is the random number generator; If None, the random number
    #: generator is the RandomState instance used by np.random.
    random_seed = models.IntegerField(
        "Random Seed",
        blank=True, null=True,
        help_text=(
            'The seed of the pseudo random number generator to use when '
            'shuffling the data. If int, random_state is the seed used by the '
            'random number generator; If RandomState instance, random_state '
            'is the random number generator; If None, the random number '
            'generator is the RandomState instance used by np.random.'
        )
    )

    #: Auto-generated Image if available
    image = models.ImageField(
        "Image",
        blank=True, null=True,
        help_text=(
            'Auto-generated Image if available'
        )
    )

    class Meta:
        verbose_name = "Support Vector Machine for Classification"
        verbose_name_plural = "Support Vector Machines for Classification"
        app_label = "supervised_learning"

    def __init__(self, *args, **kwargs):
        kwargs["sl_type"] = self.SL_TYPE_CLASSIFICATION
        super(SVC, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[SVM|C] {0}".format(self.name))

    def get_engine_object(self):
        # -> Ensure defaults
        gamma = self.kernel_coefficient
        if not gamma:
            gamma = 'auto'
        max_iters = self.engine_iterations
        if not max_iters:
            max_iters = -1
        cache_size = self.cache_size
        if not cache_size:
            cache_size = 200
        classifier = svm.SVC(
            C=self.penalty_parameter,
            kernel=self.kernel,
            degree=self.kernel_poly_degree,
            gamma=gamma,
            coef0=self.kernel_independent_term,
            shrinking=self.use_shrinking,
            probability=self.estimate_probability,
            tol=float(self.tolerance),
            cache_size=cache_size,
            class_weight=self.class_weight,
            verbose=True,  # self.verbose,
            max_iter=max_iters,
            decision_function_shape=self.decision_function_shape,
            random_state=self.random_seed
        )
        self.engine_object = classifier
        return(classifier)

    def get_conf_dict(self):
        conf_dict = {}
        conf_dict['name'] = self.name
        conf_dict['kernel'] = self.get_kernel_display()
        kernel_details = ""
        if self.kernel == "poly":
            kernel_details += "Degree: {} ".format(
                self.kernel_poly_degree)
        if (self.kernel == "poly" or self.kernel == "sigmoid" or
                self.kernel == "rbf"):
            kc = self.kernel_coefficient if self.kernel_coefficient else "Auto"
            kernel_details += "Coef. (gamma): {} - ".format(kc)
        if (self.kernel == "poly" or self.kernel == "sigmoid"):
            kit = self.kernel_independent_term \
                if self.kernel_independent_term else "0"
            kernel_details += "Indep. Term: {} - ".format(kit)
        conf_dict['kernel_details'] = kernel_details
        conf_dict['penalty_parameter'] = self.penalty_parameter
        conf_dict['str'] = "Kernel: {}{}, Penalty: {}".format(
            self.get_kernel_display(), kernel_details, self.penalty_parameter
        )
        return(conf_dict)
