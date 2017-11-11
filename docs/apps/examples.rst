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

The goal of this example is to show how to use unsupervised learning (clustering) to segmentate your users based on metrics of usage patterns.

For this, we will discuss the SmartDjango site, which is an example biased - but not limited - towards applications of e-commerce / shopping and content-based sites - like news, blogging, etc.

It can also be thought as a piece of a recommendation system, where the recommendations are based on what "similar" users (in the same group) do.

Recording the metrics
---------------------

Recording metrics is application-specific, depends on your case: how your data is already "organized" or modelled, what you are trying to measure, your goals and the tools availablity.

In this case, the SmartDjango company knows - from their experience building and running the site over the time - that there are different groups of users which consume the application in different ways: among all the pages of the site, there are intrested pages (main content) which are not equally appealing to all the users.

They want to learn how these groups "behave" in order to provide a more personalized experience to engage users - like different templates, filtering of items - or at a higher level - promotions, content, etc.

As an initial approach, starting from the scratch, the measuring of usage patterns of users should be done with the simplest metrics: how much time a user spends on a interested page (on average), and how many times does the user visits it.

The SmartDjango site is small and has about 2 hundreds of different pages (different urls for items), so they would have 400 metrics recorded for each user.

Each metric represents a dimension in the data that will feed the model, so each user usage pattern would be represented as an *R^400* point or observation. This is unsuitable for the models or techinques that `django-ai` currently support because they are affected by the `Curse of Dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`_ (you should read this `explanation for Machine Learning <https://stats.stackexchange.com/a/169351>`_).

A solution for this is recording the metrics at a higher level of agreggation, instead of a metric for each page, collect for different groups / categories / types of pages. This way, the dimensionality of the input is reduced to a "useful space" for the model.

If the SmartDjango company was a concrete news or blogging site, their interested pages would be the news or posts, which are usually already categorized with sections like "Sports" (A), "Tech" (B), "Art" (C), "Culture" (D), etc. or the "main" tag of the post.

In the case of a concrete shopping site, their interested pages could be the categories of their products, like "Shoes" (A), "Laptops" (B), "Bags" (C), etc.

For other content-based sites, the categories usually "emerge naturally", like in music or movies sites they could be the genres. In other cases, you may have to categorize them according to your goals.

In this case, SmartDjango has categorized their intrested pages according to their role, resulting in 10 categories or types of pages - "A" ... "J" - resulting in 10 metrics of the form:

``avg_time_pages_X``
    Average Time Spent on Pages of type X (Seconds).

``visits_pages_X``
    Amount of Visits on Pages of type X (Hits).

(with `X` = "A", ..., "J")

.. autofunction:: examples.metrics.metric_visits_and_avg_time_on_pages


For implementing the recording, SmartDjango had to update both front-end and back-end parts of the application.

In the front-end, the base template of the interested pages was updated to include the neccesary Javascript to measure the time spent on the page and then unobtrusively POST it to the back-end, where the implemented Metrics Pipeline executes all the code that calculates and store the metrics away from the user's "navigation" request cycle:

.. _examples_process_metrics:

.. autofunction:: examples.views.process_metrics

Realizing that the most visited ones are from type A, SmartDjango decided to go first on a smaller step, for the first building block the focus should be put on A-pages and the rest. Then, another metric should be recorded:

``avg_time_pages``
    Average Time Spent on Pages, independently of their type (Seconds).

``visits_pages``
    Amount of Visits on Pages, independently of their type (Hits).

.. autofunction:: examples.metrics.metric_visits_and_avg_time_on_pages

Constructing the model
----------------------

Averages have the "nice" property of being well modelled with Gaussian (Normal) variables (even in the odd cases when it is not normally distributed, the approximation is "good"). In order to find the groups (clusters) of users which share a common usage pattern, a mixture of Normal variables will be used with only 2 of the recorded metrics: ``avg_time_pages`` and ``avg_time_pages_a``. Note that this metrics are not independent, a "hit" on a Page of type A will also make a "hit" on Pages.

Once the groups have been identified, you can take actions based on their belonging and even find more patterns inside them.

This is an adaptation of the `BayesPy Gaussian Mixture Model Example <http://bayespy.org/examples/gmm.html>`_, where you can find more details about it, in this case there will be also 4 clusters, which can be briefly characterized by:

- A first group that stays briefly on the site, and does not care about pages of type A, they stay short independently of the page type.
- Two "opposite" "central" groups: one will stay almost the same time on pages of type A while their interest vary on other pages whereas the other has a fixed interest on other pages and a varying degree on A-types.
- A fourth group which stays longer in the site with a negative correlation: the higher they stay on pages of type A, the shorter they stay on other pages - and vice-versa. 

(if interest is well measured by time spent on it :).

You can see the details of its construction in the following migration:

.. autofunction:: examples.migrations.0005_add_avg_times_clusters.populate_avg_times

The ``Network type`` is set to "Clustering" in the Bayesian Network object and the network topology can be briefly described as:

``alpha``
    The prior for the probability of assignment to a cluster (group).
``Z``
    The belonging to a cluster (group).
``mu`` and ``Lambda``
    The mean vectors (``mu``) and precision matrices (``Lambda``) of the Gaussian clusters.
``Y``
    The mixture itself that observes the data from the users.

After the inference is performed, the model should identify the 4 real clusters from the 10 initial ones set in the uninformative priors of the models (see the nodes parameters in the admin interface).

Note that due to the random initialization of the BayesPy VB engine - due to the Categorical ``Z`` node - the algorithm may converge to local optimums which are not the global optima.

For avoiding this, the ``Engine Meta Iterations`` in the Bayesian Network object (see :ref:`bayesian_network`) is set to 15. This will repeat the inference 15 times from different states of the random number generator and choose the one with the highest likelihood. Also, ``django-ai`` conveniently takes care of doing all the re-labelling required between inferences so you can compare the results and automate the inference and the actions you might take according to the results.

You can watch this effect by running the inference several times from the admin's interface and tweaking the parameter.

By design, ``django-ai`` tries to take outside Django's request cycle as many calculations as possible. This means all the heavy stuff like model estimation is done once and stored. Inside the request cycle - i.e. in a view or a template - it is just regular queries to the database or operations that are "pretty straight-forward", like cluster assignment.

So, performance-wise, it doesn't matter if you choose 1000 meta iterations and takes an hour to complete them.

Also note that this model (Gaussians Mixture) can't "discern" when the clusters are overlapped - like the "central" "opposite" ones. For those users, it will be highly unlikely that they will be assigned to the right ones.

Once you are confident with the results, they will be stored in the ``cluster_1`` field of the ``UserInfo`` model (as it was set in the ``Results Storage`` field of the Bayesian Network object - see :ref:`bayesian_networks`). Note that the ``_1`` in the model field is to emphasize that you may end up with several clusters for the user with different metrics and it can be used as an input for other networks or techniques.

This way, they will be efficiently available in all your views and templates, so you can do things like:

.. code-block:: python

    def my_view(request):
        ...
        if request.user.user_info.main_group == "A":
            return(redirect('/deals/A'))
        # or
        products = Products.objects.filter_for_group(
                                request.user.user_info.cluster_1)
        ...

.. _examples_clustering_automation:

Automation
----------

After a while, many new users come and also usage patterns may change, resulting in a never-ending learning process. How to automate this dependes on the case, there are 2 main intended ways:

1. Through the BN internal counter and its threshold, which will be used in this case.
2. Through scheduled and background tasks - the preferred way.

Each time a metric is posted (see :ref:`above <examples_process_metrics>`) the BN counter is incremented by 1, and each time a new user is created the counter is incremented by 5. Once the counter reaches its threshold - 10 - it triggers an update of the model: it recalculates all with the new data and re-assigns the users to the new results.

These numbers are arbitrary and are intended to showcase the process. Setting the Threshold, when and how to increment the counter depends on your case. The logic here is "when 'enough' new data has arrived or the data have changed 'significantly', recalculate the model".

Also note that the Counter Threshold might trigger the update **inside** the user's "navigation" request cycle, as when creating a new user. You should realize that the process is happening when you experience the delay in the browser. This is intentional, to show what you **SHOULDN'T** do:

.. autofunction:: examples.views.new_user

In the case of processing metrics, the same code does not give any problems to the user experience because it is outside the "navigation" request cycle (see its source :ref:`above <examples_process_metrics>`). Try commenting out the update method in ``new_user`` and everything will go smoothly.

You can disable this behaviour by setting the Counter Threshold field to nothing in the admin.

Using scheduled and background tasks is the preferred way because it avoids completly the chance of messing with user's "navigation" request cycle, which may end up being detrimental to the user experience.

You can do this with apps like `Celery <http://docs.celeryproject.org/en/latest/django/first-steps-with-django.html>`_.

Updating the counter may be just a query to the database and may be not worthy of the overhead of a background task, but the model inference is indeed what you want to schedule, i.e. at midnight:

.. code-block:: python

    @periodic_task(
        run_every=(crontab(minute=0, hour=0)),
        name="recalculate_bn_if_necessary",
        ignore_result=True
    )
    def recalculate_bn_if_necessaryl(bn_name, threshold):
        """
        Recalculates the model of a BN if the data has changed enough
        """
        bn = BayesianNetwork.objects.get(name=bn_name)
        if bn.counter >= your_threshold:
            bn.perform_inference(recalculate=True)
            bn.counter = 0
            bn.save()
            logger.info(
                "BN {} had a counter of {} and has been recalculated".format(
                    bn.name, bn.counter)
            )

(You may also want to use `django-celery-beat <https://pypi.python.org/pypi/django_celery_beat>`_).

You can opt for no automation at all and "manually" recalculate all the model through the admin when deemed neccesary. This may be suitable for the beginning, but with the adequate tunning, you can build and incorporate an autonomous system that constantly learns from the data into your application. 

Other Considerations
--------------------

Each time a model recalculation is done, you may take a look at the bottom of the admin page of the BN where the graphic of the current model is shown along with clusters table to monitor it (you may need to reload the page :).

You might find that the technique - Bayesian Gaussian Mixture Model for Clustering implemented via the BayesPy's engine - is "unstable": it produces different results when the data changes a bit - and even when it doesn't change.

This posses a problem for automation and for an AI system, as decisions are taken based on labels it produces. If the labels change dramatically, those decisions (routing, filtering, content, etc.) may end up being meaningless.

There are a variety of reasons of why this happens on two different levels - the model and the implementation - from which we will review three of them.

First, the model itself features automatic selection of the number of clusters in the data.

Selecting or determining the number of clusters is a key problem in Unsupervised Learning and not an easy one. This technique "selects automatically" that number by starting from a maximum number - set in the nodes parameters of the priors and hyper-parameters - and after fitting the model, the ones that have "changed" from its "initial state" are the number of clusters in the data. This comes with a cost, it adds complexity and many chances for the optimization to "stall" in local optimums if there isn't enough data for the estimation.

If you reduce the maximum number of possible clusters, you will see an increase of the stability of the results.

The initial number of clusters to search in the data is set in the Dirichlet Node (named `alpha`) parameters (set to 10 in ``numpy.full(``\ 10\ ``, 1e-05))``) and in hyper-parameters nodes `mu` and `Lambda` plates, which models the clusters means and covariance matrices (set to 10 in ``numpy.zeros(2), [[1e-5,0], [0, 1e-5]], plates=(``\ 10\ ``, )`` and ``2,  [[1e-5,0], [0, 1e-5]], plates=(``\ 10\ ``, )``).

If you change this to 4 (the real number in this example) - or 5 for contemplating outliers (see below) - you will end up in a better shape for automating. In general, initially choose a "big number", then change to the number you expect in your data.

For seeing how more data makes the algorithm converge you can set the settings variable ``DJANGO_AI_EXAMPLES_USERINFO_SIZE``, otherwise the table size is defaulted to 200. After setting it to a different value, you have to recreate the ``UserInfo`` by issuing ``python manage.py migrate django_ai.examples zero`` and ``python manage.py migrate django_ai.examples`` to the console. It is suggested that you try with 800, 2,000 and 20,000 among your values.

Second, the Gaussian Mixture Model is not robust in the presence of outliers.

Outliers, sometimes refered as "contamination" or "noise", are atypical observations in the data that lead to incorrect model estimation (see `here <https://en.wikipedia.org/wiki/Robust_statistics>`_ for a more detailed introduction to the problem).

One solution to this is using heavy-tailed elliptical distributions in the mixture - such as Student's ´t´ - instead of Gaussians for the clusters, but supporting this would require extending the BayesPy framework and it is out of the scope of this example.

Outliers will happen in your data - as you will see, they are easily generated - adding instability to the results of the technique.

To mitigate this, add an extra cluster to number of clusters you expect. If the number of outliers or the proportion to the "normal" ones is "low enough", they will be "captured" in that extra group.

Third, optimization "stalls" in local optimums in the VB engine.

For dealing with this, do not skimp on the ``Meta Iterations`` parameter of the BN discussed previously. 

Finally - and not related to the causes - you can also improve the stability and the speed of the results by using informative priors with what you have been observing - besides low-level tunning of the engine (which is out of the scope of this example).

Seeing is Believing
--------------------

Last but not least, run the development server and you can see all of this in action by going to http://localhost:8000/examples/pages and monitor it through the admin and the console log.

At this point, you should be able to understand the workings of the tool and start making your Django application smarter :)
