.. _api:

===
API
===

``django_ai`` API is what makes possible to seamessly integrate different Statistical Models and Techniques into your Django project.

It provides an abstraction layer so you can integrate any machine learning or statistical engine to the framework and therefore to your project.

The API is designed to provide the main goal of `Interchangeabilty.`

Each Statistical Model or Technique is isolated via the API and provides its functionality through the interface. This allows to interchange any Technique within a System or your code seamessly, provided that they are of the same type (i.e. Classification techniques).

This decoupling (or pattern) has very nice consecuences, such as allowing versioning to improve the models and algorithms independently.

The Application Programming interface
-------------------------------------

Each System or Technique should be encapsulated in a Django model which inherits from the appropiate ``django_base.model``:

.. autoclass:: base.models.StatisticalModel

.. autoclass:: base.models.SupervisedLearningTechnique

.. autoclass:: base.models.UnsupervisedLearningTechnique
