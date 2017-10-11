.. _examples:

========
Examples
========

This app contains sample data and example objects of the apps of ``django-ai``.

UserInfo Model
==============

This is a generated Model that mimics information about the users of a Django project. It is a common pattern for many web applications, to record metrics about the usage of the application to identify patterns besides the information of the user available. It will provide data for the statistical modelling.

It is populated in the following way:

.. autofunction:: examples.migrations.0003_populate_userinfo.populate_userinfos

.. autofunction:: examples.migrations.0005_add_avg_times_clusters.populate_avg_times


Bayesian Network Example 1
==========================

The goal of this example is to show how to construct a very simple Bayesian Network for modelling the mean and the precision of a metric recorded for each user that we will assume that is normally distributed.

With this, the application can decide actions for the users based on the "position" of its metric in the population. Although this particular application can be done with point estimates, the objective here is to show the modelling.

This is based on the example provided in `BayesPy's Quickstart <http://bayespy.org/user_guide/quickstart.html>`_, where you can find all the math details of the model.

Once the inference is ran, you can do something like:

.. code-block:: python

    from django_ai.models.bayesian_networks import BayesianNetwork

    bn = BayesianNetwork.objects.get(name="BN1 (Example)")
    Q = bn.engine_object
    mu = Q['mu'].get_moments()[0]
    tau = Q['tau'].get_moments()[0]
    sigma = sqrt(1 / tau)
    if (request.user.avg1 < mu - 2 * sigma or
        request.user.avg1 > mu + 2 * sigma):
        print("Hmmm... the user seems to be atypical on avg1, I shall do something")


Bayesian Network Example 2
==========================

The goal of this example is to show how to use unsupervised learning (clustering) for segmentate your users based on metrics of usage patterns (recording such metrics depends on the application).

The metrics used for this will be:

``avg_time_logged``
    Average Time Logged In (Weekly)

    This metric can be obtained with tools like `django-session-log <https://github.com/agfunder/django-session-log>`_ and / or with javascript that post information to be processed.
``avg_time_pages_a``
    Average Time spent on Pages of type A (Weekly)

    This metric can be obtained with middleware that process the users' requests, in Django views or with javascript posting the information.

Averages have the "nice" property of being well modelled with Gaussian (Normal) variables (even the approximation is "good"). To find the groups (clusters) of users which share a common usage pattern, a mixture of Normal variables will be used.

Once the groups have been identified, you can take actions based on their belonging and even find more patterns inside them.

This is an adaptation of the `BayesPy Gaussian Mixture Model Example <http://bayespy.org/examples/gmm.html>`_, where you can find more details about it, but in short the Nodes that will be used are:

``alpha``
    The prior for the probability of assignment to a cluster.
``Z``
    The belonging to a cluster.
``mu`` and ``Lambda``
    The mean vectors (``mu``) and precision matrices (``Lambda``) of the Gaussian clusters.
``Y``
    The mixture of itself.

Once the inference is performed, the model identifies 4 clusters from the 10 initial ones.

The next step is assigning the users to their clusters:

.. code-block:: python

    from django_ai.models.bayesian_networks import BayesianNetwork

    clustering_bn = BayesianNetwork.objects.get(name="Clustering (Example)")
    Z = clustering_bn.engine_object["Z"]
    # Get the cluster labels
    zm = Z.get_moments()
    labels = []
    for z in zm[0]:
        labels.append(numpy.argmax(z))
    # Save them
    for index, userinfo in enumerate(UserInfo.objects.all()):
        userinfo.user.group = label
        userinfo.user.save(update_fields["group"])

Then it will be efficiently available on all your views and templates:

.. code-block:: python

    def my_view(request):
        ...
        if user.group == 3:
            return(redirect('/deals/3'))
        products = Products.objects.filter_for_group(request.user.group)
        ...

As new users comes, once the metrics are available, you can assing him / her to a group:

.. code-block:: python

    def update_user_group(sender, **kwargs):
        user_info = kwargs['instance'] 
        user_info.user.update_group(user_info.group_metrics())

    post_save.connect(update_user_group, sender=UserInfo)

After a while, many new users comes and also usage patterns may change, you can recalculate the model from the admin or programatically:

.. code-block:: python

    clustering_bn = BayesianNetwork.objects.get(name="Clustering (Example)")
    clustering_bn.perform_inference(recalculate=True)

with the caveat that you might need to do some re-labelling (according to the cluster means).

These snippets gives you the idea of how you can use the results of the model in your application.

You can automate many of this tasks conveniently with apps like `Celery <http://docs.celeryproject.org/en/latest/django/first-steps-with-django.html>`_.
