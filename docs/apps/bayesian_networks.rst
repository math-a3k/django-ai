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

This is the main object for a Bayesian Network.

It gathers all Nodes and Edges of the DAG that defines the Network.

All the results of the inference will be available here and this object is what you will be using inside the code.

The following fields are shown:

``Name``
    The name of the Bayesian Network. This must be an unique identifier, meant to be used for retrieving the object (i.e. ``BayesianNetwork.objects.get(name='BN 3 - Final'``)

``Engine Object Timestamp``
    This is an auto-generated field, timestamping the last inference done or `None` if not available.

``Image``
    This is an auto-generated field, shown at the bottom of the page. It will be updated each time a Node or an Edge is added or modified to the Network.

``Engine Meta Iterations``
    Runs the Inference Engine (BayesPy's VB) *N* times and picks the result with the highest likelihood. This is only useful when a Node in the Network requires random initialization (see :ref:`custom_keywords`), as the algorithm may converge to a local optimum. Otherwise, it will repeat the result *N* times. It defaults to 1. 

``Engine Iterations``
    The maximum number of iterations of the Inference Engine (`BayesPy's VB update method <http://bayespy.org/user_api/generated/generated/generated/bayespy.inference.VB.update.html#bayespy.inference.VB.update>`_). It defaults to 100. 

Bayesian Network Node
---------------------

Each BayesPy Node in the Network is represented here.

Nodes can be either `Stochastic` or `Deterministic`.

`Stochastic` reffers to representing a Random Variable, `Deterministic` reffers to representing a transformation or function on other Nodes. 

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

It is designed to be a just like the ``*args`` and ``**kwargs`` you pass to a function or method programatically with some restrictions, described below:

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

    In some ocassions, there must be a reference to a function instead of the result of it, this is done by preceding the function with an ``@``, i.e. ``@bayespy.nodes.Gaussian()`` will return the function object (in this case the whole class) instead of the result of it.

    Due to security reasons, the allowed namespaces must be specified in a list named ``DJANGO_AI_WHITELISTED_MODULES`` in your settings, i.e.::

        DJANGO_AI_WHITELISTED_MODULES = ['numpy', 'bayespy.nodes', 'scipy']

    By default, only ``numpy`` and ``bayespy.nodes`` are enabled.

For example, the string::
  
  True, 2, 1e-6, mu, numpy.ones(2), [[1,2], [3,4]], type=rect, sizes=[3, 4,], coords = ([1,2],[3,4]), func=numpy.zeros(2)

will be equivalent to doing programatically::
  
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
    Triggers the deletion of a keyword argument. Use this when you do not want a keyword argument to be set automatically. Currently, only ``plates`` is set automatically for Stochastic Observable Nodes when it is not specified and it is set to the "shape" of the data being observed. In some types of networks this can interfere with `BayesPy` plates propagation. To avoid this, use ``:noplates`` in the Node's parameters.

``:ifr``
    Triggers Initializate from Random in the Node's engine object.

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

Edges are neccesary for resolving dependencies between nodes, i.e. if Node takes another Node as a parameter, there must be an Edge between the Nodes so the Child is able to access its Parents.

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

    It will initialize or create all the BayesPy Nodes, initialize the Inference Engine of BayesPy and perform the inference with it. Once is run, it will save all the results in the Bayesian Network object and the corresponding in each Node, generating the Node image where corresponds and updating the timestamps. See the API section for accesing the results. 

``Reset inference on the network``
    This will reset (set to `None`) all the engine- and inference-related fields in the network.


API
===

For integrating the objects into your code, you simply have to import the Django model whenever deemed neccesary and get the network you want to use:

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

If you want to do things programatically, you should see this migration of the ``examples`` app:

.. autofunction:: examples.migrations.0004_bn_example.create_bn1_example

and take a look at the :download:`tests <../../tests/test_bns.py>`.

.. _`bayespy-quickstart`: http://bayespy.org/user_guide/quickstart.html
