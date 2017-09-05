=============================
django-ai
=============================

.. image:: https://badge.fury.io/py/django-ai.svg
    :target: https://badge.fury.io/py/django-ai

.. image:: https://travis-ci.org/math-a3k/django-ai.svg?branch=master
    :target: https://travis-ci.org/math-a3k/django-ai

.. image:: https://codecov.io/gh/math-a3k/django-ai/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/math-a3k/django-ai

Artificial Intelligence for Django
==================================

``django-ai`` is a collection of apps for integrating statistical models into your
Django project so you can implement machine learning conveniently.

It integrates several libraries and engines providing your Django app with a set of 
tools so you can leverage the data generated in your project. 

.. image:: /django_ai.jpg
    :target: https://django-ai.readthedocs.io/en/latest/overview.html

Documentation
-------------

The full documentation is at https://django-ai.readthedocs.io or the `/docs` directory for offline reading.

Quickstart
----------

The easiest way of trying `django-ai` is inside its package:

1. Create a virtual environment and activate it::
    
    python3 -m venv django-ai_env
    source django-ai_env/bin/activate

2. Install django-ai::
    
    (django-ai_env) pip install django-ai

3. Change into the `django-ai` directory::

    (django-ai_env) cd django-ai_env/lib/python3.5/site-packages/django_ai

4. Create the migrations for the dependencies and apply them::
    
    python manage.py makemigrations django_dag
    python manage.py migrate

5. Create a superuser::
    
    python manage.py createsuperuser

6. Start the development server and visit http://127.0.0.1:8000/admin/ to look at the examples and start creating your statistical models::

    python manage.py runserver

Or you can clone it from the repository and install the requirements in a virtualenv::

    git clone git@github.com:math-a3k/django-ai.git

and do the same steps, installing the requirements in a virtual
environment from ``requirements.txt``

For installing it in your project, please refer `here <https://django-ai.readthedocs.io/en/latest/installation.html>`_.


Features
--------

* Bayesian Networks: Integrate Bayesian Networks through your models using the BayesPy framework.

See the `Overview <https://django-ai.readthedocs.io/en/latest/overview.html>`_ section in the documentation for more inforation.

Running Tests
-------------

Does the code actually work?

::

    source <YOURVIRTUALENV>/bin/activate
    (myenv) $ pip install tox
    (myenv) $ tox

Credits
-------

Tools used in rendering this package:

*  Cookiecutter_
*  `cookiecutter-djangopackage`_

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`cookiecutter-djangopackage`: https://github.com/pydanny/cookiecutter-djangopackage
