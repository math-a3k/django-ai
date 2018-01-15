.. _api:

===
API
===

``django_ai`` API is what makes possible to seamessly integrate different Statistical Models and Techniques into your Django project.

It provides an abstraction layer so you can integrate any machine learning or statistical engine to the framework and therefore into your project.

Design Goals
============

The API is designed to provide the main goals of `Interchangeabilty`, `Persistance across requests` and `Separation from the User's request cycle`.

Interchangeabilty
-----------------

Each Statistical Model or Technique implemented by an engine is isolated via the API and provides its functionality through the interface. This allows to interchange any Technique within a System or your code seamessly - provided that they are of the same type (i.e. Classification techniques).

This decoupling (or pattern) has very nice consecuences, such as allowing versioning to improve the models and algorithms independently.

Persistance across requests
---------------------------

The inference, calculations, state, etc. done by the engines must be available and persisted across requests.

Separation from the User's request-response cycle
-------------------------------------------------

Heavy calculations should be taken away from the User's request cycle, done independently and expose the relevant results in it.

The Application Programming interface
=====================================

There are three main types of cases that ``django_ai`` provides: Systems, General Techniques and Techniques.

Systems use Techniques or General Techniques to make a full implementation of a Statistical Model on a particular task, providing "end-to-end functionality" - like a Spam Filter. They provide an API on their own, besides leveraging the API of Techniques.

General Techniques are those which provide a framework for implementing Techniques - like Bayesian Networks. They are like a System but they do not provide "end-to-end functionality" but building blocks instead.

Techniques are particular implementations of a Statistical Model providing the building block for constructing higher functionality.

Each one should be encapsulated in a Django model which inherits from the appropiate class from ``django_ai.base.models``, providing all the functionality through the public API.

.. _api_statistical_model:

Statistical Model API
---------------------

This is the base class for all, it defines the basic functionality that must be adhered to comply with the Design Goals of the API - besides each System or Technique implementing particular fields and methods for their functioning.

General Techniques should subclass this Abstract Model and implement the required methods. 

.. autoclass:: base.models.StatisticalModel

General
^^^^^^^

Fields 
++++++

.. autoattribute:: base.models.StatisticalModel.name

    The name is meant to be the way the object is retrieved / referenced from other Techniques, Systems or plain code, so, it must be unique.

.. autoattribute:: base.models.StatisticalModel.sm_type

.. autoattribute:: base.models.StatisticalModel.SM_TYPE_CHOICES

.. autoattribute:: base.models.StatisticalModel.metadata

    This field for storing results and information related to internal tasks (pre-processing, visualization, etc.). It is initialized on ``.save()``

Methods
+++++++

.. automethod:: base.models.StatisticalModel.rotate_metadata


Engine-related
^^^^^^^^^^^^^^

The Engine-related fields and methods are those that encapsulates a Statistical Engine and provide a uniform interface for the framework.

A Statistical Engine is the third-party implementation of an algorithm or technique which is being integrated, such as *BayesPy* or *scikit-learn*.

Fields
++++++

.. autoattribute:: base.models.StatisticalModel.engine_object

.. autoattribute:: base.models.StatisticalModel.engine_object_timestamp

.. autoattribute:: base.models.StatisticalModel.engine_meta_iterations

    In the cases where the Statistical Engine uses random initialization in the algorithm for performing inference, the result may depend on that initial state. If the engine does not provide a solution for this, a way of improving this is to run the Engine inference several times (``Meta Iterations``) and, given a measure of "improvement", choose the best result.

    The measure of improvement and the selection of the result must be implemented in the ``perform_inferece`` method.

.. autoattribute:: base.models.StatisticalModel.engine_iterations

    The Statistical Engines usually provide a safeguard parameter to set the Maximum Iterations for the case when the convergence of the optimization method or algorithm is not guarranted or to avoid excessive run-time in some setups.

.. autoattribute:: base.models.StatisticalModel.is_inferred

Methods
+++++++

.. automethod:: base.models.StatisticalModel.get_engine_object

.. automethod:: base.models.StatisticalModel.reset_engine_object

.. automethod:: base.models.StatisticalModel.perform_inference

.. automethod:: base.models.StatisticalModel.reset_inference


.. _api_automation:

Automation
^^^^^^^^^^

For simple automation of the System or Technique into a project, three fields are provided: ``Counter``, ``Counter Threshold`` and ``Threshold Actions``.

.. autoattribute:: base.models.StatisticalModel.counter
.. autoattribute:: base.models.StatisticalModel.counter_threshold
.. autoattribute:: base.models.StatisticalModel.threshold_actions

The rationale is very simple: increment the counter until a threshold where actions are triggered.

Is up to the user when, where and how the counter is incremented. If the field ``Counter Threshold`` is set, when this counter reaches that Threshold, the actions in ``Threshold Actions`` will be run on the object's ``save()`` method or the evaluation can be triggered with the following method:

.. automethod:: base.models.StatisticalModel.parse_and_run_threshold_actions

**IMPORTANT**: The user should take care also to avoid triggering ``Threshold Actions`` inside of the user's "navigation" request-response cycle, which may lead to hurt the user experience. For a concrete example, see :ref:`here <examples_clustering_automation>`.

The allowed keywords for ``Threshold Actions`` are set in Model constant ``ACTIONS_KEYWORDS``:

.. autoattribute:: base.models.StatisticalModel.ACTIONS_KEYWORDS

which defaults to:

    ``:recalculate``
        Recalculates (performs again) the inference on the model.

.. _api_supervised_learning:

Supervised Learning Technique
-----------------------------

This is the Base Class for `Supervised Learning <https://en.wikipedia.org/wiki/Supervised_learning>`_ Techniques and Systems.

Besides having all the functionality of :ref:`api_statistical_model`, it defines the common interface for Supervised Learning.

.. autoclass:: base.models.SupervisedLearningTechnique

General
^^^^^^^

Fields 
++++++

.. autoattribute:: base.models.SupervisedLearningTechnique.sl_type
    :annotation: Supervised Learning Type

.. autoattribute:: base.models.SupervisedLearningTechnique.SL_TYPE_CHOICES

Engine-related
^^^^^^^^^^^^^^

Fields
++++++

.. autoattribute:: base.models.SupervisedLearningTechnique.cv_is_enabled
    
    `Cross Validation (CV) <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ is model validation technique, its goal is to estimate the performance of a predictive model in an independent dataset.

.. autoattribute:: base.models.SupervisedLearningTechnique.cv_folds

.. autoattribute:: base.models.SupervisedLearningTechnique.cv_metric

Methods
+++++++

.. automethod:: base.models.SupervisedLearningTechnique.perform_cross_validation

Data-related
^^^^^^^^^^^^

Fields
++++++

.. autoattribute:: base.models.SupervisedLearningTechnique.labels_column

.. autoattribute:: base.models.SupervisedLearningTechnique.pretraining

Methods
+++++++

.. automethod:: base.models.SupervisedLearningTechnique.get_labels

.. automethod:: base.models.SupervisedLearningTechnique.get_pretraining_data

.. automethod:: base.models.SupervisedLearningTechnique.get_pretraining_labels

.. _api_unsupervised_learning:

Unsupervised Learning Technique
-------------------------------

This is the Base Class for `Unsupervised Learning <https://en.wikipedia.org/wiki/Unsupervised_learning>`_ Techniques and Systems.

Besides having all the functionality of :ref:`api_statistical_model`, it defines the common interface for Unsupervised Learning.

.. autoclass:: base.models.UnsupervisedLearningTechnique
