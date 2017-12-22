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
        ('poly', "Polynomic"),
        ('rbf', "RBF"),
        ('linear', "Linear"),
        ('sigmoid', "Sigmoid"),
        ('precomputed', "Precomputed"),
    )
    # C : float, optional (default=1.0)
    penalty_parameter = models.FloatField(
        "Penalty Parameter",
        default=1.0, blank=True, null=True,
        help_text=(
            'Penalty parameter C of the error term.'
        )
    )

    # kernel : string, optional (default=’rbf’)
    kernel = models.CharField(
        "SVM Kernel",
        choices=SVM_KERNEL_CHOICES, blank=True, null=True, max_length=50,
        help_text=(
            'It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, '
            '‘precomputed’ or a callable. If none is given, ‘rbf’ will'
            ' be used. If a callable is given it is used to pre-compute '
            'the kernel matrix from data matrices; that matrix should be '
            'an array of shape (n_samples, n_samples).'
        )
    )

    # degree : int, optional (default=3)
    kernel_poly_degree = models.IntegerField(
        default=3, blank=True, null=True,
        help_text=(
            'Degree of the polynomial kernel function (‘poly’). Ignored '
            'by all other kernels.'
        )
    )

    # gamma : float, optional (default=’auto’)
    kernel_coefficient = models.FloatField(
        "Kernel coefficient (fixed)",
        blank=True, null=True,
        help_text=(
            'Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. '
            'Leave blank "for automatic" (1/n_features will be used)'
        )
    )

    # coef0 : float, optional (default=0.0)
    kernel_independent_term = models.FloatField(
        default=0.0, blank=True, null=True,
        help_text=(
            'Independent term in kernel function. It is only significant '
            'in "poly" and "sigmoid".'
        )
    )

    # probability : boolean, optional (default=False)
    estimate_probability = models.BooleanField(
        default=False,
        help_text=(
            'Whether to enable probability estimates. This must be '
            'enabled prior to calling fit, and will slow down that method'
        )
    )

    # shrinking : boolean, optional (default=True)
    use_shrinking = models.BooleanField(
        default=True,
        help_text=(
            'Whether to use the shrinking heuristic.'
        )
    )

    # tol : float, optional (default=1e-3)
    tolerance = models.FloatField(
        default="1e-3", blank=True, null=True,
        help_text=(
            "Tolerance for stopping criterion."
        )
    )

    # cache_size : float, optional
    cache_size = models.FloatField(
        'Kernel Cache Size (MB)',
        blank=True, null=True,
        help_text=('Specify the size of the kernel cache (in MB).'),
    )

    # class_weight : {dict, ‘balanced’}, optional
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

    # verbose : bool, default: False
    verbose = models.BooleanField(
        'Verbose',
        default=False,
        help_text=(
            'Enable verbose output. Note that this setting takes advantage '
            'of a per-process runtime setting in libsvm that, if enabled, '
            'may not work properly in a multithreaded context.'
        ),
    )

    # Already in engine_meta_iterations
    # max_iter : int, optional (default=-1)
    # Hard limit on iterations within solver, or -1 for no limit.

    # decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
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

    # random_state : int, RandomState instance or None, optional (default=None)
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
    image = models.ImageField(
        "Image",
        blank=True, null=True,
        help_text=(
            'Auto-generated Image if available'
        )
    )

    class Meta:
        verbose_name = "Support Vector Machine - Classification"
        verbose_name_plural = "Support Vector Machines - Classification"
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
        kernel = self.kernel
        if not kernel:
            kernel = 'rbf'
        max_iters = self.engine_iterations
        if not max_iters:
            max_iters = -1
        cache_size = self.cache_size
        if not cache_size:
            cache_size = 200
        classifier = svm.SVC(
            C=self.penalty_parameter,
            kernel=kernel,
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
