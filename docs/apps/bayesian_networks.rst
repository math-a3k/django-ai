.. _bayesian_networks:

=================
Bayesian Networks
=================

This app provides `Bayesian Network modelling <https://en.wikipedia.org/wiki/Bayesian_network>`_ through integrating the BayesPy framework: http://bayespy.org.

If you are not familiar with the framework, it is better at least take a glance on its `excellent documentation <http://bayespy.org/user_guide/quickstart.html>`_ for a better understanding on how the modelling is done.


Front-end
=========

All the configuration should be done through the admin of Bayesian Networks - or more specifically, through the `change form`.

Bayesian Network
----------------

This is the main object for a Bayesian Network.

It gathers all nodes and edges of the DAG that defines the network.

All the results of the inference will be available here and this object is what you will be using inside the code.

The following fields are shown:

``Name``
	The name of the Bayesian Network. This must be an unique identifier, meant to be used for retrieving the object (i.e. ``BayesianNetwork.objects.get(name='BN 3 - Final'``)

``Engine Object Timestamp``
	This is an auto-generated field, timestamping the last inference done or `None` if not available.

``Image``
	This is an auto-generated field, shown at the bottom of the page. It will be updated each time a Node or an Edge is added or modified to the network.

Bayesian Network Node
---------------------

Each BayesPy Node in the network is represented here.

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
	The Parameters for the Distribution of the Node. They must be separated by a comma and a space ("``, ``") and currently only names (strings) or scalars (numbers) are supported. If it is a name, it must be the name of another Node in the network which you are passing to the Node.

	The Parameters must be according to the Distribution chosen, otherwise the initialization of the BayesPy Node will fail. For a list of the Distribution Parameters see the `BayesPy documentation <http://bayespy.org/user_api/generated/bayespy.nodes.html#stochastic-nodes>`_.

``Is Observable``
	If the Random Variable is observable or not. If it is observable, then it will need to be linked to a field of a Django model where the data will be held, set in the ``Reference model`` and ``Reference column`` fields.

``Reference Model``
	The Django model that will held the data in the case of an Observable Stochastic Node.

``Reference Column``
	The name of the field in the Django model that will held the data in the case of an Observable Stochastic Node. It must be the name "callable" attribute.

Deterministic Fields
^^^^^^^^^^^^^^^^^^^^

``Deterministic``
	The function or transformation that the Node applies

``Deterministic Params``
	The Parameters for the function of the Node. They must be separated by a comma and a space ("``, ``") and currently only names (strings) or scalars (numbers) are supported. If it is a name, it must be the name of another Node in the network which you are passing to the Node.

	The Parameters must be according to the Deterministic Node chosen, otherwise the initialization of the BayesPy Node will fail. For a list of the Deterministic Parameters see the `BayesPy documentation <http://bayespy.org/user_api/generated/bayespy.nodes.html#deterministic-nodes>`_.

Visualization
^^^^^^^^^^^^^

``Graph Interval``
	(`Stochastic only`) Depending on the Distribution, a graphic may be available. This is the graphing interval, separated by a comma and a space, i.e. "``-10, 20``".

``Image``
	(`Stochastic only`) This is an auto-generated field, once the inference is run on the network, if it is available, an image with the graph of the distribution of the Random Variable will be stored here and shown at the bottom of the page.

Timestamps
^^^^^^^^^^

``Engine Object Timestamp``
	The Timestamp of the BayesPy Node creation

``Engine Inferred Object Timestamp``
	The Timestamp of the last inference on the Node or the network.


Bayesian Network Edge
---------------------

Each Edge between Nodes in the Direct Acyclic Graph of the Bayesian Network is represented here.

The following fields are shown:

``Parent``
	The "From" Node

``Child``
	The "To" Node

``Description``
	A brief description of the Edge (i.e. "``mu -> tau``")

Actions
-------

The main are:

``Run inference on the network``
	This will run the Inference on the current state of the network. It will initialize or create all the BayesPy Nodes, initialize the Inference Engine of BayesPy and perform the inference with it. Once is run, it will save all the results in the Bayesian Network object and the corresponding in each Node, generating the Node image where corresponds and updating the timestamps. See the API section for accesing the results. 

``Reset inference on the network``
	This will reset (set to `None`) all the engine- and inference-related fields in the network.


API
===

For integrating the objects into your code, you simply have to import the Django model whenever deemed neccesary and get the network you want to use:

.. code-block:: python

	from django_ai.models.bayesian_networks import BayesianNetwork

	bn = BayesianNetwork.objects.get(name="<NAME OF MY BN>")

If the network is inferred, the results of it, the ``VB`` object is stored in the ``engine_object`` field. This is a BayesPy object which you can use at your will:

.. code-block:: python

    Q = bn.engine_object
	mu = Q['mu'].get_moments()[1]
	tau = Q['tau'].get_moments()[1]
	sigma = sqrt(1 / tau)
	if (request.user.avg1 < mu - 2 * sigma or
	    request.user.avg1 > mu + 2 * sigma):
	    print("Hmmm... the user seems to be atypical on avg1, I shall do something")

If you want to do things programatically, you should see this migration of the ``examples`` app:

.. literalinclude:: ../../django_ai/examples/migrations/0004_bn_example.py

and take a look at the :download:`model definition <../../django_ai/bayesian_networks/models.py>`
