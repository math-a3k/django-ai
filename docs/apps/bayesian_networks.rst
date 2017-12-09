.. _bayesian_networks:

=================
Bayesian Networks
=================

This app provides `Bayesian Network modelling <https://en.wikipedia.org/wiki/Bayesian_network>`_ through integrating the BayesPy framework: http://bayespy.org.

If you are not familiar with the framework, it is better at least take a glance on its `excellent documentation <bayespy-quickstart_>`_ for a better understanding on how the modelling is done.


Front-end
=========

All the configuration should be done through the admin of Bayesian Networks - or more specifically, through the `change form`.

.. _bayesian_network:

Bayesian Network
----------------

This is the main object for a Bayesian Network (BN).

It gathers all Nodes and Edges of the DAG that defines the Network.

All the results of the inference will be available here and this object is what you will be using inside the code.

The following fields are available for configuration:

``Name``
    The name of the Bayesian Network. This must be an unique identifier, meant to be used for retrieving the object (i.e. ``BayesianNetwork.objects.get(name='BN 3 - Final')``)

``Network Type``
    The type of the Network. Based on this field, the internal methods of the Bayesian Network object perform different actions. Currently, there are 2 Types:

    ``General``
        Performs the inference with the BayesPy engine on the Bayesian Network and set the resulting object in the ``engine_object`` field.

    ``Clustering``
        Besides performing the inference with the BayesPy engine (and setting the result in the ``engine_object`` field), it performs tasks like cluster re-labelling, process the results and stores useful information in the ``metadata`` field.

        It assumes that the Network topology is from a Gaussian Mixture Model with:
            - One Categorical Node for clusters assigments depending on a Dirichlet Node for prior probabilities;
            - One Gaussian Node for clusters means and a Wishart Node for clusters covariance matrices;
            - One Observable Mixture Node for the observations.

        Other topologies are not supported at the moment.

``Results Storage``
    In the case of networks which have a labelling output - such as Clustering or Classification - this sets where to store the results for convenience. It must have the following syntax: ``<storage>:params``.

    The following storages are available:

    ``dmf``
        *Django Model Field*: Saves the results to a field of a Django model. The model should be accessible by Django's Content Types framework and - **IMPORTANT**: it uses its default order provided by the model manager for storing. That ordering should be the same as the data retrieved by :ref:`bayesian_networks_node_column`, otherwise "manual" storing should be done for your situation.

        Its parameters are a dotted path: ``<app_label>.<model>.<field>``,  i.e. ``dmf:examples.UserInfo.cluster_1`` will store the results to the ``cluster_1`` field of the ``UserInfo`` model.

Miscellaneous Fields
^^^^^^^^^^^^^^^^^^^^

``Engine Meta Iterations``
    Runs the Inference Engine (BayesPy's VB) *N* times and picks the result with the highest likelihood. This is only useful when a Node in the Network requires random initialization (see :ref:`custom_keywords`), as the algorithm may converge to a local optimum. Otherwise, it will repeat the result *N* times. It defaults to 1.

``Engine Iterations``
    The maximum number of iterations of the Inference Engine (`BayesPy's VB update method's repeat <http://bayespy.org/user_api/generated/generated/generated/bayespy.inference.VB.update.html#bayespy.inference.VB.update>`_). It defaults to 1000.

``Counter``
    Internal Counter of the Bayesian Networks meant to be used in automation. Is up to the user to increment the counter when deemed necessary. If the field ``Counter Threshold`` is set, when this counter reaches that Threshold, the actions in ``Threshold Actions`` will be run on the object's ``save()`` method or the evaluation can be triggered with the following method:

    .. automethod:: bayesian_networks.models.BayesianNetwork.parse_and_run_threshold_actions

    **IMPORTANT**: As it is up to the user when, where and how the counter is incremented, the user should take care also to avoid triggering ``Threshold Actions`` inside of the user's "navigation" request cycle, which may lead to hurt the user experience. For a concrete example, see :ref:`here <examples_clustering_automation>`.

``Counter Threshold``
    Threshold of the Internal Counter, meant to be used in automation. If this field is not set, the ``Threshold Actions`` will not be triggered on the object's ``save()`` method.

``Threshold Actions``
    Actions to be ran when the Internal Counter reaches or surpasses the Counter Threshold, evaluated on model's ``save()``. The actions must be specified by keywords separated by spaces. Currently, the supported keywords are:

    ``:recalculate``
        Recalculates (performs again) the inference on the Network.

``Engine Object Timestamp``
    This is an auto-generated field, timestamping the last inference done or `None` if not available.

``Image``
    This is an auto-generated field, shown at the bottom of the page. It will be updated each time a Node or an Edge is added or modified to the Network.

``Metadata``
    This is an internal field for storing results and information related to internal tasks (pre-processing, visualization, etc.). It is shown here for convenience as its content may be used for integrating the Bayesian Network into the application's code.

Bayesian Network Node
---------------------

Each BayesPy Node in the Network is represented here.

Nodes can be either `Stochastic` or `Deterministic`.

`Stochastic` refers to representing a Random Variable, `Deterministic` refers to representing a transformation or function on other Nodes.

Each type of Nodes have a fields associated with it, which will need to be filled accordingly.

General Fields
^^^^^^^^^^^^^^

``Name``
    The name of the Node. This must be an unique identifier inside the network. It will be used for passing the Node to other Nodes as a parameter among others.

``Node Type``
    If the Node is `Stochastic` or `Deterministic`. This determines which fields will be taken into account for Node creation.

Stochastic Fields
^^^^^^^^^^^^^^^^^

``Distribution``
    The Distribution of the Node

``Distribution Params``
    The Parameters for the Distribution of the Node. See :ref:`node_parameters` below for more details.

    The Parameters must be according to the Distribution chosen, otherwise the initialization of the BayesPy Node will fail. For a list of the Distribution Parameters see the `BayesPy documentation for Stochastic Nodes <http://bayespy.org/user_api/generated/bayespy.nodes.html#stochastic-nodes>`_.

``Is Observable``
    If the Random Variable is observable or not. If it is observable, then it will need to be linked to fields or callables of a Django Model where the data will be held, set in :ref:`bayesian_networks_node_column`.

Deterministic Fields
^^^^^^^^^^^^^^^^^^^^

``Deterministic``
    The function or transformation that the Node applies.

``Deterministic Params``
    The Parameters for the function of the Node. See :ref:`node_parameters` below for more details.

    The Parameters must be according to the Deterministic Node chosen, otherwise the initialization of the BayesPy Node will fail. For a list of the Deterministic Parameters see the `BayesPy documentation for Deterministic Nodes <http://bayespy.org/user_api/generated/bayespy.nodes.html#deterministic-nodes>`_.


.. _node_parameters:

Node Parameters
^^^^^^^^^^^^^^^

The string set in the ``Distribution Params`` and ``Deterministic Params`` fields is parsed and used for initialization of BayesPy Nodes.

It is designed to be a just like the ``*args`` and ``**kwargs`` you pass to a function or method programmatically with some restrictions, described below:

``Booleans and Keywords``
    ``True``, ``False``, ``None``.
``Scalars``
    Integers and Floats, i.e ``1, -2, -0.3, 1e-06``.
``Structures``
    Lists and Tuples, i.e. ``[1, 2], [[1e-06, 2], [3, 4]], (2, 3,), ([1, 2], [3, 4])``.
``Strings``
    Strings are reserved for Node names. To pass another Node as a parameter to it simply use its name. Nodes' names are resolved through Network Edges of the graph (see :ref:`bayesian_networks_edge`).
``Custom Keywords``
    Strings starting with ``:`` - i.e. ``:no`` are considered as "Custom Keywords" for ``django-ai`` and triggers different behaviours. See :ref:`custom_keywords`.
``Functions``
    Functions *must be namespaced* and their arguments can be anything of the above.

    In some occasions, there must be a reference to a function instead of the result of it, this is done by preceding the function with an ``@``, i.e. ``@bayespy.nodes.Gaussian()`` will return the function object (in this case the whole class) instead of the result of it.

    Due to security reasons, the allowed namespaces must be specified in a list named ``DJANGO_AI_WHITELISTED_MODULES`` in your settings, i.e.::

        DJANGO_AI_WHITELISTED_MODULES = ['numpy', 'bayespy.nodes', 'scipy']

    By default, only ``numpy`` and ``bayespy.nodes`` are enabled.

For example, the string::
 
  True, 2, 1e-6, mu, numpy.ones(2), [[1,2], [3,4]], type=rect, sizes=[3, 4,], coords = ([1,2],[3,4]), func=numpy.zeros(2)

will be equivalent to doing programmatically::
 
  MyNode(True, 2, 1e-6, mu, numpy.array([ 1.,  1.], [[1,2], [3,4]], type=rect, sizes=[3, 4,], coords = ([1,2],[3,4]), func=numpy.array([ 0.,  0.])

With this, a ``GaussianARD`` Node can be initialized with::

  mu, tau

where ``mu`` and ``tau`` are parents Nodes, or for a 2D ``Gaussian`` Node::
 
  numpy.ones(2), numpy.zeros(2)


.. _custom_keywords:

Custom Keywords
~~~~~~~~~~~~~~~

Node parameters' strings starting with ``:`` are considered *Custom Keywords*, they should be used at an ``*arg`` level and their meaning or behaviour triggered is described below:

``:noplates``
    Triggers the deletion of the ``plates`` keyword argument. Use this when you do not want a keyword argument to be set automatically. Currently, ``plates`` is only set automatically for Stochastic Observable Nodes when it is not specified and it is set to the "shape" of the data being observed. In some types of networks this can interfere with `BayesPy` plates propagation. To avoid this, use ``:noplates`` in the Node's parameters.

``:ifr``
    Triggers `Initialize from Random` in the Node's engine object.

``:dl|<NODE_NAME>``
    Uses the Data Length of of Node ``NODE_NAME``. Meant to be used in plates, i.e. ``plates=(:dl|Y, )``


Visualization
^^^^^^^^^^^^^

``Graph Interval``
    (`Stochastic only`) Depending on the Distribution, a graphic may be available. This is the graphing interval, separated by a comma and a space, i.e. "``-10, 20``".

``Image``
    (`Stochastic only`) This is an auto-generated field, once the inference is run on the network, if it is available, an image with the graph of the distribution of the Random Variable will be stored here and shown at the bottom of the page.

Timestamps
^^^^^^^^^^

``Engine Object Timestamp``
    The Timestamp of the BayesPy Node creation.

``Engine Inferred Object Timestamp``
    The Timestamp of the last inference on the Node or the network.


.. _bayesian_networks_node_column:

Bayesian Network Node Column
----------------------------

In the case of Stochastic Observable Nodes, an inline will be displayed for setting the columns / axis / dimensions that will represent the observations of the Random Variable of the Node.

There is no restrictions on number of columns nor they should be on the same Django model, only that they must contain the same amount of records / size.

The following fields are shown:  

``Reference Model``
    The Django Model that will held the data.

``Reference Column``
    The name of the field or attribute in the Django model that will held the data.

``Position``
    The ordering of the columns, set automatically by the nested inline.


.. _bayesian_networks_edge:

Bayesian Network Edge
---------------------

Each Edge between Nodes in the Direct Acyclic Graph of the Bayesian Network is represented here.

Edges are necessary for resolving dependencies between nodes, i.e. if Node takes another Node as a parameter, there must be an Edge between the Nodes so the Child is able to access its Parents.

The following fields are shown:

``Parent``
    The "From" Node.

``Child``
    The "To" Node.

``Description``
    A brief description of the Edge (i.e. "``mu -> tau``").

Actions
-------

The main are:

``Run inference on the network``
    This will run the Inference on the current state of the network.

    It will initialize or create all the BayesPy Nodes, initialize the Inference Engine of BayesPy, perform the inference with it and the appropriate tasks corresponding to the `Network Type`. Once is run, it will save all the results in the Bayesian Network object and the corresponding in each Node, generating the Node image where corresponds and updating the timestamps. See the API section for accesing the results.

``Reset inference on the network``
    This will reset (set to `None`) all the engine- and inference-related fields in the network.

``Re-initialize the random number generator``
    This will reinitialize Python's random number generator. For unknown reasons yet, sometimes the Inference Engine gets stuck, re-initializing the RNG and resetting the inference may solve the issue without restarting the server.


API
===

For integrating the objects into your code, you simply have to import the Django model whenever deemed necessary and get the network you want to use:

.. code-block:: python

    from django_ai.models.bayesian_networks import BayesianNetwork

    bn = BayesianNetwork.objects.get(name="<NAME-OF-MY-BN>")

If the network is inferred, the results of it - the ``VB`` object - is stored in the ``engine_object`` field. This is a BayesPy object which you can use at your will:

.. code-block:: python

    Q = bn.engine_object
    mu = Q['mu'].get_moments()[1]
    tau = Q['tau'].get_moments()[1]
    sigma = sqrt(1 / tau)
    if (request.user.avg1 < mu - 2 * sigma or
        request.user.avg1 > mu + 2 * sigma):
        print("Hmmm... the user seems to be atypical on avg1, I shall do something")

You can perform all the Actions on the Network with the following methods of the ``BayesianNetwork`` objects:

.. autoclass:: bayesian_networks.models.BayesianNetwork
   :members: get_engine_object, perform_inference, reset_inference

If you want to do things programmatically, you should see the migrations of the ``examples`` app:

.. autofunction:: examples.migrations.0004_bn_example.create_bn1_example
.. autofunction:: examples.migrations.0006_clustering_bn_example.create_clustering_bn_example

and take a look at the :download:`tests <../../tests/apps/bayesian_networks/test_bns.py>`.

.. _`bayespy-quickstart`: http://bayespy.org/user_guide/quickstart.html
