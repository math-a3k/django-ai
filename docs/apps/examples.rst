.. _examples:

========
Examples
========

This app contains sample data and example objects of the apps of ``django-ai``.

UserInfo Model
==============

This is an generated Model that mimics information about the users of a Django project. It is a common pattern for many web applications. It will provide data for the statistical modelling.

It is constructed in the following way:

.. literalinclude:: ../../django_ai/examples/migrations/0003_populate_userinfo.py

Bayesian Network Example
========================

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
