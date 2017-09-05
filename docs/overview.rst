.. _overview:

========
Overview
========

``django-ai`` is a collection of apps for integrating statistical models into your Django project so you can implement machine learning conveniently.

It integrates several libraries and engines providing your Django app with a set of tools so you can leverage the data generated in your project.

The integration is done through Django models, where most of the data is generated and stored in a Django project. This data is what will feed the statistical models and then those models will be used in the project's code.

The rationale of ``django-ai`` is to provide for each statistical model or technique bundled, a front-end for configuration and an API for integrating it into your code.

The front-end aims to let you choose which parts of the Django models will be used and "configure" the parameters of it: "state" the statistical model. Currently, it is admin-based.

Once you are "happy" with your model you can incorporate it into your code, i.e. in a view:

.. code-block: python
    from django_ai.apps.bayesian_networks.models import BayesianNetwork

    def my_view(request):
        user_classifier = BayesianNetwork.objects.get(name="User BN")
        user_class = user_classifier.predict(request.user)
        # Do something with the user classification
        return redirect(route_user(user_class))

This "kind of hybrid approach" is what gives you the convenience: you can state the model easily in the front-end (admin), and after you incorporate it in your code, you can mantain, improve or update it in the front-end.

``django-ai`` aims to provide with 2 classes of apps or statistical models: "low level" and "high level".

"Low level" are those "basic" models or techniques, such as Bayesian Networks, Support Vector Machines, Classification and Aggregation Trees, Random Forests, Clustering, Neural Networks, etc. Those are the building blocks for the machine to learn and construct its intelligence.

"High level" are those which are composed from "low level" ones, such as a recommender system.

The apps will integrate the models already implemented in other libraries as much as possible, using them as engines and becoming also a front-end to them for Django projects.

The primary or main integration is with Python codebases - for obvious reasons. In the future integration with other codebases is in sight, such as ``R`` where the amount of statistical models implemented is the biggest or ``Haskell`` where the purity of the languange makes it ideal for expressing mathematical models.

This is an Overview of the Philosophy, Design, Architecture and Roadmap of ``django-ai``.

Currently it is in ``alpha`` stage, your are welcome to join the community of users and developers.

Last but not least: ``django-ai`` is, and will always be, Free Software (Free as in Freedom). If you can't patent Math, you can't patent Software. Did Newton hide something from you? :) Open Knowledge is better for all :)