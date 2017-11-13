.. :changelog:

=======
History
=======

0.0.1 (2017-11-13)
++++++++++++++++++

django-ai 0.0.1 Release Notes
-----------------------------

I'm very happy to announce the first release of django-ai: Artificial Intelligence for Django!!

``django-ai`` is a collection of apps for integrating statistical models into your Django project so you can implement machine learning conveniently.

It aims to integrate several libraries and engines providing your Django app with a set of tools so you can leverage your project functionality with the data generated within.

The integration is done through Django models - where most of the data is generated and stored in a Django project - and an API focused on integrating seamlessly within Django projects’ best practices and patterns.

The rationale of ``django-ai`` is to provide for each statistical model or technique bundled a front-end for configuration and an API for integrating it into your code.

Excited?

- :ref:`overview`
- :ref:`quickstart`
- :ref:`examples` 

You are welcome to join the community of users and developers :)

Features
~~~~~~~~

* Bayesian Networks: Integrate Bayesian Networks through your models using the `BayesPy framework <http://bayespy.org/>`_.

Known Issues
~~~~~~~~~~~~

* In development mode (``DEBUG = True``) the BayesPy Inference Engine may stall during model estimation on certain states of the Pseudo Random Number Generator. You may need to reset the PRNG or deactivate and activate again your Python virtualenv. This does not affect other operations like cluster assigment.

0.0.1a0 (2017-08-31)
++++++++++++++++++++

* First release on PyPI.

