.. _overview:

========
Overview
========

``django-ai`` is a collection of apps for integrating statistical models into your Django project so you can implement machine learning conveniently.

It aims to integrate several libraries and engines providing your Django app with a set of tools so you can leverage your project functionality with the data generated within.

The integration is done through Django model - where most of the data is generated and stored in a Django project - and an API focused on integrating seamlessly within Django projects’ best practices and patterns. This data is what will feed the statistical models and then those models will be used in the project's code to augment its utility.

The rationale of ``django-ai`` is to provide for each statistical model or technique bundled a front-end for configuration and an API for integrating it into your code.

The front-end aims to let you choose which parts of the Django models will be used and "configure" its the parameters: conveniently "state" the statistical model. Currently, it is admin-based.

Once you are "happy" with your model you can incorporate it into your code, i.e. in a view:

.. code-block:: python

    from django_ai.apps.bayesian_networks.models import BayesianNetwork

    def my_view(request):
        user_classifier = BayesianNetwork.objects.get(name='User BN')
        user_class = predict(user_classifier, request.user)
        # Do something with the user classification
        return(redirect('promotions', user_class))

This "kind of hybrid approach" is what gives you convenience: you can state the model easily in the front-end (admin), and after you incorporate it in your code, you can maintain, improve or update it in the front-end.

By design, ``django-ai`` tries to take outside Django's request cycle as many calculations as possible. This means all the heavy stuff like model estimation is done once and stored. Inside the request cycle - i.e. in a view or a template - it is just regular queries to the database or operations that are "pretty straight-forward", like cluster assignment or classification.

See :ref:`examples` for more.

``django-ai`` aims to provide with 2 classes of apps or statistical models: "low level" and "high level".

"Low level" are those "basic" models or techniques, such as Bayesian Networks, Support Vector Machines, Classification and Aggregation Trees, Random Forests, Clustering algorithms, Neural Networks, etc. Those are the building blocks for the machine to learn and construct its intelligence.

"High level" are those which are composed from "low level" ones, such as a recommender system or a spam filter.

The ``django-ai`` apps will integrate the models already implemented in other libraries as much as possible, using them as engines and becoming also a front-end to them for Django projects.

The primary or main integration is with Python codebases - for obvious reasons. In the future, integration with other codebases is in sight, such as ``R`` where the amount of statistical models implemented is the biggest or ``Haskell`` where the purity of the language makes it ideal for expressing mathematical models.

This is an Overview of the Philosophy, Design, Architecture and Roadmap of ``django-ai``.

You are welcome to join the community of users and developers.

Last but not least: ``django-ai`` is, and will always be, Free Software (Free as in Freedom). If you can't patent Math, you can't patent Software. Did Newton hide something from you? :) Open Knowledge is better for all :)
