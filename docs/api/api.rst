.. _api:

===
API
===

``django_ai`` API is what makes possible to seamessly integrate different Statistical Models and Techniques into your Django project.

It provides an abstraction layer so you can integrate any machine learning or statistical engine to the framework and therefore to your project.

The API is designed to provide the main goal of `Interchangeabilty`.

Each Statistical Model or Technique is isolated via the API and provides its functionality through the interface. This allows to interchange any Technique within a System or your code seamessly, provided that they are of the same type (i.e. Classification techniques).

This decoupling (or pattern) has very nice consecuences, such as allowing versioning to improve the models and algorithms independently.

The Application Programming interface
=====================================

There are two main types of cases that ``django_ai`` provides: Systems & General Techniques and Techniques.

Systems use Techniques to make a full implementation of a Statistical Model providing "end-to-end functionality" - like a Spam Filter - and provide an API on their own and besides leveraging on the API of the Techniques.

General Techniques are those that provide a framework for implementing Techniques - like Bayesian Networks. They are a System but they do not provide "end-to-end functionality".

Techniques are particular implementations of a Statistical Model providing the building blocks for constructing higher functionality.

Each one should be encapsulated in a Django model which inherits from the appropiate class from ``django_ai.base.models``, providing all the functionality through the public API.

Systems and "General" Techniques
--------------------------------

.. autoclass:: base.models.StatisticalModel

Fields
^^^^^^

Besides particular fields that each System or Technique implement for their functioning, these common fields allow the "plugging" or the "interchangeabilty".

.. autoattribute:: base.models.StatisticalModel.name

General fields
++++++++++++++


``Metadata``
    This is an internal field for storing results and information related to internal tasks (pre-processing, visualization, etc.). It is shown here for convenience as its content may be used for integrating the Bayesian Network into the application's code.


Engine-related
++++++++++++++

.. autoattribute:: base.models.StatisticalModel.engine_object

.. autoattribute:: base.models.StatisticalModel.engine_object_timestamp

.. autoattribute:: base.models.StatisticalModel.engine_meta_iterations

In the cases where the Statistical engine uses random initialization of the algorithm for performing inference of the Technique and the result depends on that initial state. If the engine does not provide a solution for this, a way of improving this is to run the Engine inference several times (``Meta Iterations``) and, given a measure of "improvement", choose the best result. 

The measure of improvement must be implemented in the ``perform_inferece`` method.

.. autoattribute:: base.models.StatisticalModel.engine_iterations

The Statistical engines usually provide a safeguard parameter to set the Maximum Iterations for the case when the convergence of the optimization method or algorithm is not guarranted or to avoid excessive run-time in some setups.


Automation
++++++++++

For simple automation of the System or Technique into a project, three fields are provided: ``Counter``, ``Counter Threshold`` and ``Threshold Actions``.

.. autoattribute:: base.models.StatisticalModel.counter
.. autoattribute:: base.models.StatisticalModel.counter_threshold
.. autoattribute:: base.models.StatisticalModel.threshold_actions

The rationale is very simple: increment the counter until a threshold where actions are triggered.

Is up to the user when, where and how the counter is incremented. If the field ``Counter Threshold`` is set, when this counter reaches that Threshold, the actions in ``Threshold Actions`` will be run on the object's ``save()`` method or the evaluation can be triggered with the following method:

.. automethod:: base.models.StatisticalModel.parse_and_run_threshold_actions

**IMPORTANT**: The user should take care also to avoid triggering ``Threshold Actions`` inside of the user's "navigation" request cycle, which may lead to hurt the user experience. For a concrete example, see :ref:`here <examples_clustering_automation>`.

The allowed keywords for ``Threshold Actions`` are set in Model constant ``ACTIONS_KEYWORDS``:

.. autoattribute:: base.models.StatisticalModel.ACTIONS_KEYWORDS

which defaults to:

	``:recalculate``
	    Recalculates (performs again) the inference on the Network.

Supervised Learning Technique
-----------------------------

.. autoclass:: base.models.SupervisedLearningTechnique

Unsupervised Learning Technique
-------------------------------

.. autoclass:: base.models.UnsupervisedLearningTechnique
