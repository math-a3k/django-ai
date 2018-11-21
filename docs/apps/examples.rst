.. _examples:

========
Examples
========

This app contains sample data and example objects of the apps of ``django-ai``.


UserInfo Model
==============

This is a generated Model that mimics information about the users of a Django project. It is common for many web applications to record metrics about the usage of the application to identify patterns besides the information of the user available. It will provide data for the statistical modelling.

It is mainly populated in the following way:

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


.. _clustering_bn_example:

Clustering with Bayesian Networks (Example 2)
=============================================

The goal of this example is to show how to use unsupervised learning (clustering) to segmentate your users based on metrics of usage patterns.

For this, we will discuss the SmartDjango site, which is an example biased - but not limited - towards applications of e-commerce / shopping and content-based sites - like news, blogging, etc.

It can also be thought as a piece of a recommendation system, where the recommendations are based on what "similar" users - in the same group - do.


Recording the metrics
---------------------

Recording metrics is application-specific, depends on your case: how your data is already "organized" or modelled, what you are trying to measure, your goals and the tools' availability.

In this case, the SmartDjango company knows - from their experience building and running the site over the time - that there are different groups of users which consume the application in different ways: among all the pages of the site, there are pages (content) which are not equally appealing to all the users.

They want to learn how these groups "are" in order to provide a more personalized experience to engage users - like different templates, filtering of items - or at a higher level - promotions, content, etc.

As an initial approach, starting from the scratch, the measuring of usage patterns of users should be done with the simplest metrics: how much time a user spends on a interested page (on average), and how many times does the user visits it.

The SmartDjango site is small and has about 2 hundreds of different pages (different urls for items or content items), so 400 metrics would have to be recorded for each user.

Each metric represents a dimension in the data that will feed the model, so each user usage pattern would be represented as an *R^400* point or observation. This is unsuitable for the models or techinques that we are using in this example - a Bayesian Network of Gaussian Mixtures - because they are affected by the `Curse of Dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`_ (you should read this `explanation for Machine Learning <https://stats.stackexchange.com/a/169351>`_).

A solution for this is recording the metrics at a higher level of aggregation, instead of a metric for each page, collect for different groups / categories / types of pages. This way, the dimensionality of the input is reduced to a "useful space" for the model.

This showcase the need for "aligning" the model and the data in order to have a succesful application of the learning into the project. Having chosen a model for a goal, the data must in line with it so the solution it produces is the best within its scope or limitations - every model trades generality for acurracy in some way, for getting better in particular. This model works - it has been already proven that it will produce optimal results (in some sense) given cerntain conditions (characteristics of the input / data), having them "aligned" is what makes the best out of the tool.

If the SmartDjango company was a concrete news or blogging site, their interested pages would be the news or posts, which are usually already categorized with sections like "Sports" (A), "Tech" (B), "Art" (C), "Culture" (D), etc. or the "main" tag of the post.

In the case of a concrete shopping site, their interested pages could be the categories of their products, like "Shoes" (A), "Laptops" (B), "Bags" (C), etc.

For other content-based sites, the categories usually "emerge naturally", like in music or movies sites they could be the genres. In other cases, you may have to categorize them according to your goals.

In this case, SmartDjango has categorized their interested pages according to their role, resulting in 10 categories or types of pages - "A" ... "J" - resulting in 10 metrics of the form:

``avg_time_pages_X``
    Average Time Spent on Pages of type X (Seconds).

``visits_pages_X``
    Amount of Visits on Pages of type X (Hits).

(with `X` = "A", ..., "J")

.. autofunction:: examples.metrics.metric_visits_and_avg_time_on_pages

For implementing the metrics recording, SmartDjango had to update both front-end and back-end parts of the application.

In the front-end, the base template of the interested pages was updated to include the necessary Javascript to measure the time spent on the page and then unobtrusively POST it to the back-end, where the implemented Metrics Pipeline executes all the code that calculates and store the metrics away from the user's "navigation" request cycle:

.. _examples_process_metrics:

.. autofunction:: examples.views.process_metrics

Realizing that the most visited ones are from type A, SmartDjango decided to go first on a smaller step: for the first building block of the system the focus should be put on "A"-pages and then, the rest. Therefore, another metric should be recorded:

``avg_time_pages``
    Average Time Spent on Pages, independently of their type (Seconds).

``visits_pages``
    Amount of Visits on Pages, independently of their type (Hits).

.. autofunction:: examples.metrics.metric_visits_and_avg_time_on_pages


Constructing the model
----------------------

The reason for choosing recording averages is that they have the "nice" property of being well modelled with Gaussian (Normal) variables - even in the "odd" cases when it is not normally distributed, it takes a factible amount of measurements in this case - time spent by users - for the approximation to be "good".

So, given that we will be dealing with averages, we will use a model that is geared towards that.

In order to find the groups (clusters) of users which share a common usage pattern, a mixture of Normal variables will be used with only 2 of the recorded metrics: ``avg_time_pages`` and ``avg_time_pages_a``. Note that this metrics are not independent, a "hit" on a Page of type A will also make a "hit" on Pages.

Once the groups have been identified, you can take actions based on their belonging and even find more patterns inside them using "extra information" (other metrics gathered not included for this model).

This is an adaptation of the `BayesPy Gaussian Mixture Model Example <http://bayespy.org/examples/gmm.html>`_, where you can find more details about it, in this case there will be also 4 clusters, which can be briefly characterized by:

- A first group that stays briefly on the site, and does not care about pages of type A, they stay short independently of the page type.
- Two "opposite" "central" groups: one will stay almost the same time on pages of type A while their interest vary on other pages whereas the other has a fixed interest on other pages and a varying degree on A-types.
- A fourth group which stays longer in the site with a negative correlation: the higher they stay on pages of type A, the shorter they stay on other pages - and vice-versa.

(if interest is well measured by time spent on it :).

You can see the details of its construction in the following migration:

.. autofunction:: examples.migrations.0005_add_avg_times_clusters.populate_avg_times

For implementing the model, the ``Network type`` is set to "Clustering" in the Bayesian Network object and the network topology can be briefly described as:

``alpha``
    The prior for the probability of assignment to a cluster (group).
``Z``
    The belonging to a cluster (group).
``mu`` and ``Lambda``
    The mean vectors (``mu``) and precision matrices (``Lambda``) of the Gaussian clusters.
``Y``
    The mixture itself that observes the data from the users.

After the inference is performed, the model should identify the 4 real clusters from the 10 initial ones set in the uninformative priors of the models (see the nodes parameters in the admin interface).

This showcase how this technique chooses the number of clusters.

A problem with this technique is relying on random initialization (Variational Bayes engine) and the algorithm may converge to local optimums which are not the global optima.

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

You can opt for no automation at all and "manually" recalculate all the model through the admin when deemed necessary. This may be suitable for the beginning, but with the adequate tuning, you can build and incorporate an autonomous system that constantly learns from the data into your application.

For the cases where there is not possible or feasible to install an app like ``Celery``, with the mentioned caveats you can implement the functionality using the internal counters.

Other Considerations
--------------------

Each time a model recalculation is done, you may take a look at the bottom of the admin page of the BN where the graphic of the current model is shown along with clusters table to monitor it (you may need to reload the page :).

You might find that the technique - Bayesian Gaussian Mixture Model for Clustering implemented via the BayesPy's engine - is "unstable": it produces different results when the data changes a bit - and even when it doesn't change.

This poses a problem for automation and for an AI system, as decisions are taken based on labels it produces. If the labels change dramatically, those decisions (routing, filtering, content, etc.) may end up being meaningless.

There are a variety of reasons of why this happens on two different levels - the model and the implementation - from which we will review three of them.

First, the model itself features automatic selection of the number of clusters in the data.

Selecting or determining the number of clusters is a key problem in Unsupervised Learning and not an easy one. This technique "selects automatically" that number by starting from a maximum number - set in the nodes parameters of the priors and hyperparameters - and after fitting the model, the ones that have "changed" from its "initial state" are the number of clusters in the data. This comes with a cost, it adds complexity and many chances for the optimization to "stall" in local optimums if there isn't enough data for the estimation.

If you reduce the maximum number of possible clusters, you will see an increase of the stability of the results.

If you are on a quick read and haven't got to the details yet, the initial number of clusters to search in the data is set in the Dirichlet Node (named `alpha`) parameters (set to 10 in ``numpy.full(``\ *10* \ ``, 1e-05))``) and in hyper-parameters nodes `mu` and `Lambda` plates, which models the clusters means and covariance matrices (set to 10 in ``numpy.zeros(2), [[1e-5,0], [0, 1e-5]], plates=(``\ *10* \ ``, )`` and ``2,  [[1e-5,0], [0, 1e-5]], plates=(``\ *10* \ ``, )``).

If you change this to 4 (the real number in this example) - or 5 for contemplating outliers (see below) - you will end up in a better shape for automating. In general, initially choose a "big number", then change to the number you expect in your data.

For seeing how more data makes the algorithm converge you can set the settings variable ``DJANGO_AI_EXAMPLES_USERINFO_SIZE``, otherwise the table size is defaulted to 200. After setting it to a different value, you have to recreate the ``UserInfo`` by issuing ``python manage.py migrate django_ai.examples zero`` and ``python manage.py migrate django_ai.examples`` to the console. It is suggested that you try with 800, 2,000 and 20,000 among your values.

Second, the Gaussian Mixture Model is not robust in the presence of outliers.

Outliers, sometimes referred as "contamination" or "noise", are atypical observations in the data that lead to incorrect model estimation (see `here <https://en.wikipedia.org/wiki/Robust_statistics>`_ for a more detailed introduction to the problem).

One solution to this is using heavy-tailed elliptical distributions in the mixture - such as Student's ´t´ - instead of Gaussians for the clusters, but supporting this would require extending the BayesPy framework and it is out of the scope of this example.

Outliers will happen in your data - as you will see, they are easily generated - adding instability to the results of the technique.

To mitigate this, add an extra cluster to number of clusters you expect. If the number of outliers or the proportion to the "normal" ones is "low enough", they will be "captured" in that extra group.

Third, optimization "stalls" in local optimums in the VB engine.

For dealing with this, do not skimp on the ``Meta Iterations`` parameter of the BN discussed previously.

Finally - and not related to the causes - you can also improve the stability and the speed of the results by using informative priors with what you have been observing - besides low-level tuning of the engine (which is out of the scope of this example).

.. _bn_seeing_is_believing:

Seeing is Believing
--------------------

Last but not least, run the development server and you can see all of this in action by going to http://localhost:8000/django-ai/examples/pages and monitor it through the admin and the console log.


.. _example_spam_filtering:

Spam Filtering with SVM (Example 3)
===================================

The goal of this example is to show how to integrate the Spam Filtering system into your project and apps.

For that, we will discuss briefly the model that the SmartDjango company has implemented.

Understanding the Bag of Words projection
-----------------------------------------

As in the :ref:`previous example <clustering_bn_example>`, when SmartDjango decided to record the metrics on a per-category basis instead of per-page, the decision was made in order to reduce the dimensionality so the available toolkit can handle the problem (segmentate the users based on usage patterns).

Usually, all the Statistical Models in Machine Learning algorithms handle "numerical" inputs - or numerical representations of them, and the input in this case are texts. Strings have "natural" numerical representations in computers, like the ASCII codes (or more modern, UTF-8) internal representation, where "Hola!" is represented with (72, 111, 108, 97, 33) - a point in a 128^5 space, analogous to an R^5 point.

One of the problems with this representation is that it may be very hard to discern between similar observations in the original domain: a Natural Language.

For example, the following 4 strings are represented in the same "ASCII space" as:

``Hola!``
    ( 72, 111, 108,  97,  33,  32,  32,  32,  32,  32)
``Adios!``
    ( 65, 100, 105, 111, 115,  33,  32,  32,  32,  32)
``     HOLA!``
    ( 32,  32,  32,  32,  32,  72,  79,  76,  65,  33)
``Hola!HOLA!``
    ( 72, 111, 108,  97,  33,  72,  79,  76,  65,  33)

The first one and the third one represents the "same" in the original space (they semantically mean the same) while the second is the opposite. But, looking only at their ASCII representation (the R^10 points), the first one seems more like the second and opposite to the third one, which doesn't seem helpful for the matter.

Instead, if you consider a higher aggregation level such as words instead of characters - just like categories of pages to single pages - you may represent them (discarding the case, punctuation et al.) in:

``Hola!``
    (1, 0)
``Adios!``
    (0, 1)
``     HOLA!``
    (1, 0)
``Hola!HOLA!``
    (2, 0)

which is analogous to an R^2 point which also represents better the "structure" in the original domain at glance: the first and the third one (semantically equivalent) are represented in the same way, the fourth is in the same axis and the second is represented in another direction. Also, in a much lower complexity space.

For the task of classifying - discerning between "types" of - documents / texts / strings / messages / comments, it seems easier to work in a Word space than in a Character space.

That is the `Bag of Word representation <https://en.wikipedia.org/wiki/Bag-of-words_model>`_, which can be seen as a non-linear projection or transformation from the character space (the strings) into the word space.

In the ASCII representation, each component of the vector point (axis of the space) represents each character in the string in an arbitrary meaningless order for the domain (Natural Language) - 32 is a space, 33 is an exclamation mark, 65 is an "a", etc. Then, if the max size of the string / message is 2000 chars, we have R^2000 points.

In the Bag of Words representation, the base of the space is the set of words in the corpus (the random sample of texts, the observed messages in the database / Django model :), i.e. in the previous example, ``Hola! Adios!`` has the coordinates (1, 1) in the base {``hola``, ``adios``} and has a more meaningful order in the space for the problem.

This "better" representation provides a more suitable input for the tools available and make them perform better - i.e. achieve greater accuracy.

But, unlike the ASCII representantion where the dimension could be fixed, the dimension in this transformation is random: depending on the amount and values of the observations you have - the size and content of the corpus - the resulting dimension of the space.

As the sample size increases - the corpus has more documents - it is more likely to have more words to consider, and thus, tends to increase the dimension of the space. Once the sample is "big enough", the increase starts to lower as there are "enough" words in the base to represent new documents. How much depends on many factors, for exemplifying, a corpus of 3672 emails produces a "raw" dimension of approximately 50,000, while a corpus of 1956 comments produces points of R^5773.

As you may note, dimension can skyrocket and that generally poses a problem. Using domain-knowledge one can mitigate this, like removing the stop and common words between documents, "trimming" axes which won't contribute much to the goal of the task of classifying, providing better results in the process.

Stop words - articles ("The", "A", "An"), conjunctions (i.e. "For", "And", "But"), etc. - are common words which usually does not help to discern between document classes as they are mostly common to all sample points, so filtering them mitigates the dimensionality without affecting the classification performance. A way to filter this is either by "hard-coding" then for the Natural Language of the corpus (i.e. English), or more generally by setting a threshold in the frequency of the term, i.e. the words that appear in at least 85% percent of the documents are probably stop words (independently of the Natural Language) and won't help in discerning classes between them.

In the same reasoning, removing the terms with less frequency will help to reduce the dimensionality while trimming possible outliers than may affect the performance.

In the opposite way - but with the same goal - is instead of using one word per element of the base of the space, use two - or more - words. This is known as the `N-gram representation <https://en.wikipedia.org/wiki/N-gram>`_. The rationale of this representation is to retain better the underlying structure of the documents by constructing the base as the combinations of two (or *N*) words of the total of the corpus (vocabulary). The "better" representation has the "side-effect" of increasing the dimensionality drastically, i.e. a corpus of 1962 elements with a vocabulary of 5771 (and the same dimension for unigrams), leads to a dimension of 42,339 if you consider up to trigrams. This might impact the performance of the classifier, so it has to be balanced according to your case.

Many classifiers - including SVM - are not scale invariant, so it is recommended to remove the scale of the data - normalize or standardize it. A way of achieving this is with the `tdif-idf <https://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_ transformation, which also has the benefits of revealing stop words - among others.

Once all the texts / messages are suitably represented to be an input of the classifier of choice - the Supervised Learning Technique - the discerning between SPAM and HAM can be carried out.


Setting and Plugin the Spam Filter
----------------------------------

The SmartDjango Company have two sources of Spam: the yet-to-be-launched Comments System for their content pages and the "Contact" Forms.

These are slightly different, the comments are usually shorter than the content submitted by the forms - which seemed more like emails. Technically, one could say that it is reasonable to assume that they are generated from different stochastic processes. So, each one would have a similar but different model, different Spam Filters with their parameters tuned accordingly. We will focus on the Comment System, as the other one is analogous.

For this, minimal changes must be done at the codebase level.

The first step is making the Django model which stores the comments an Spammable Model by inherit from *IsSpammable* instead of the "traditional" Django's ``models.Model``::

    from django_ai.systems.spam_filtering.models import IsSpammable

    class CommentsOfMySite(IsSpammable):
        ...

and define two class constants in it: the name of the Spam Filter to be used and the name of the field that may contain Spam::

    class CommentsOfMySite(IsSpammable):
        SPAM_FILTER = "Comment Spam Filter (Example)"
        SPAMMABLE_FIELD = "comment"
        ...

and that's it: all objects in the Django model will be used as a source of data and new ones (created) "will go through" the Spam Filter named ``Comment Spam Filter (Example)``:

.. autoclass:: examples.models.CommentOfMySite

Then, the remaining can be done from the admin front-end: creating an Spam Filter object in the with that name.

On the admin's *change form* of the Spam Filter, they choose to use the Spammable Model they had just made available and then save it.

After, a Classifier to train and use for discerning between the Spam and Ham comments must be created. As it is high dimensional data from the Bag of Words representation, Support Vector Machines would be an adequate choice for the task.

`Support Vector Machines (SVM) <https://en.wikipedia.org/wiki/Support_vector_machine>`_ is one of the best understood (theoretically) techniques available that deals with high dimensional data with a superb performance - in terms of speed / resources and accuracy. 

They opened a new tab and created a new *Support Vector Machine for Classification* object from the `django-ai`\ 's *Supervised Learning* section in the admin. Having given a meaningful name ("SVM for Comments Spam") for it, they chose a linear kernel and small penalty parameter to start with, save it, and back to the Spam Filter, where they chose to use this classifier.

Once the classifier is set, the next is enabling metrics to evaluate its performance.

`Cross Validation (CV) <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ is a model validation technique, its goal is to estimate the performance of a predictive model in an independent dataset (i.e. in practice :).

The available CV is stratified *k*-folded, where the training sample is divided into *k* parts or folds (stratums with the same distribution of classes). Then it will iterate on each part / fold / stratum. For each fold, it will train the model with the other *k - 1* folds and test it with the current fold, recording a metric of performance (i.e. accuracy). After this, *k* metrics of performance will be available, the mean of these metrics with its standard deviation is what will be used as an estimation for the performance of the Spam Filter.

As the Comment Systems is yet to be launched, there is no data available to train the statistical model, so the SmartDjango team "went to the Internet" to see if they can find some data that could pre-train the model, so when it launches it can be fully functional while the data arrives (users commenting their content).

They found two datasets that would be useful for this purpose: the `Youtube Spam Collection <https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection>`_ and the one of the `Enron Spam Datasets <http://www2.aueb.gr/users/ion/data/enron-spam/>`_.

The first one is a collection of 1956 comments of 5 popular `YouTube <https://youtube.com>`_ videos, from which roughly the half (1005) are Spam.

From inspecting the data (available in the admin) one could tell that it may not be exactly what the comments in the SmartDjango website would look like - and that anything similar to "Subscribe to my channel" would be labelled as Spam by a model trained in this data - but it is a good starting point while the "final" data is generated.

The second one is a set of 5172 emails from the now-defunct American Corporation `Enron <https://www.google.com/search?q=enron>`_, from which roughly a third (1500) are Spam.

This will be useful for the Contact Form Spam Filter - whose content is more similar to emails. Although it has been pre-processed, the most important part of the content - the words - are there to train the model.

Recapping:

- Choose the source of data - the Spammable Django Model,
- Choose the classifier - Support Vector Machines in this case,
- Choose a Django model that contains the pre-training set - YouTube comments,
- Choose how to perform the Cross Validation are set

Once all this objects are created and set in the Spam Filter object, the process of tuning it begins.

At the bottom of the page (the Spam Filter admin's change form), each time an inference is run from the "Actions" section, the table summarizing the results of the inference is updated, showing the "Current Inference" and the "Previous Inference" performance.

Given the introduction to the Bag of Words projection, you should be able to tune its parameters to provide the classifier with a good input so it can perform best - using the Cross Validation metric to measure it.
Once you have reached a "reasonable" Bag of Words representation, it's time to tune the classifier. In this case, it will be the SVM penalty parameter (*C*) and the kernel. *Tunning C and the kernel is the key to achieve accurracy*. 

Iterating on this will produce the best parameters of the model for your data - which in this case is "pretrained" with the Youtube comments dataset.

After this, your Spam Filter is ready :)


Other Considerations
--------------------

As the Spam Filter is a subclass of :ref:`Statistical Model <api_statistical_model>`, it supports :ref:`simple automation <api_automation>`. You should take into account what has been discussed in the :ref:`previous example <examples_clustering_automation>`.

.. autoclass:: examples.views.CommentsOfMySiteView

Discussing SVMs is left out from this example for brevity. It can deal with high-dimensional data seamlessly, however, you should try to find the smallest dimension input where the data is "best separable" while maximizing its performance, which degrades with the addition of "noisy dimensions" despite of being "resistant" to overfitting and high dimensional spaces.


.. _svm_seeing_is_believing:

Seeing is Believing
--------------------

Last but not least, run the development server and you can see all of this in action by going to http://localhost:8000/django-ai/examples/comments and monitor it through the admin and the console log.
