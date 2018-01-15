.. _supervised_learning:

===================
Supervised Learning
===================

This app provides `Supervised Learning <https://en.wikipedia.org/wiki/Supervised_learning>`_ techniques for integrating them into systems or directly to your code.

From an API point of view, each technique is a particular implementation of :ref:`api_supervised_learning`.

.. _svm:

Support Vector Machines (SVM)
=============================

`Support Vector Machines <https://en.wikipedia.org/wiki/Support_vector_machine>`_ are provided by integrating the *scikit-learn* framework: http://scikit-learn.org.

If you are not familiar with the framework, it is better at least take a glance on its `excellent documentation for the technique <http://scikit-learn.org/stable/modules/svm.html#svm>`_ for a better understanding on how the modelling is done.

An example of integrating SVM into a system can be found in :ref:`example_spam_filtering`.

SVM for Classification
----------------------

All the configuration can be done through the admin of Support Vector Machines for Classification - or more specifically, through the `change form`.

The following fields are available for configuration:

General
^^^^^^^

General fields (like ``Name``) and Miscellanous are documented in the :ref:`api_statistical_model`.

This technique extends it with the following field:

.. autoattribute:: supervised_learning.models.svm.SVC.image
    :annotation: Image

The implementation uses *scikit-learn* as Engine, there is no need of setting more than 1 ``Engine Meta Iterations``.

Model Parameters
^^^^^^^^^^^^^^^^
.. autoattribute:: supervised_learning.models.svm.SVC.kernel
    :annotation: SVM Kernel
.. autoattribute:: supervised_learning.models.svm.SVC.penalty_parameter
    :annotation: Penalty parameter (C) of the error term.
.. autoattribute:: supervised_learning.models.svm.SVC.kernel_poly_degree
    :annotation: Polynomial Kernel degree
.. autoattribute:: supervised_learning.models.svm.SVC.kernel_coefficient
    :annotation: Kernel coefficient
.. autoattribute:: supervised_learning.models.svm.SVC.kernel_independent_term
    :annotation: Kernel Independent Term
.. autoattribute:: supervised_learning.models.svm.SVC.class_weight
    :annotation: Class Weight

Implementation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoattribute:: supervised_learning.models.svm.SVC.decision_function_shape
    :annotation: Decision Function Shape
.. autoattribute:: supervised_learning.models.svm.SVC.estimate_probability
    :annotation: Estimate Probability?
.. autoattribute:: supervised_learning.models.svm.SVC.use_shrinking
    :annotation: Use Shrinking Heuristic?
.. autoattribute:: supervised_learning.models.svm.SVC.tolerance
    :annotation: Tolerance
.. autoattribute:: supervised_learning.models.svm.SVC.cache_size
    :annotation: Kernel Cache Size (MB)
.. autoattribute:: supervised_learning.models.svm.SVC.random_seed
    :annotation: 
.. autoattribute:: supervised_learning.models.svm.SVC.verbose
    :annotation: Be Verbose?
