# -*- coding: utf-8 -*-

import os
from importlib import import_module
from collections import Counter

import matplotlib
matplotlib.use('Agg')

from graphviz import Digraph
import numpy as np
import bayespy as bp

from django.db import models
from django.core.files.base import ContentFile
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone
from django.db.models.signals import post_save
from django.conf import settings
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import ValidationError

from picklefield.fields import PickledObjectField
from jsonfield import JSONField

from .utils import (parse_node_args, )
if 'DJANGO_TEST' in os.environ:
    from django_ai.bayesian_networks import bayespy_constants as bp_consts
else:
    from bayesian_networks import bayespy_constants as bp_consts


class BayesianNetwork(models.Model):
    """
    Main object of a Bayesian Network.

    It gathers all Nodes and Edges of the DAG that defines the Network and
    provides an interface for performing and resetting the inference and
    related objects.
    """
    _alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _eo_meta_iterations = {}

    TYPE_GENERAL = 0
    TYPE_CLUSTERING = 1

    NETWORK_TYPE_CHOICES = (
        (TYPE_GENERAL, "General"),
        (TYPE_CLUSTERING, "Clustering"),
    )

    ACTIONS_KEYWORDS = [":recalculate", ]

    name = models.CharField("Name", max_length=100)
    image = models.ImageField("Image", blank=True, null=True)
    engine_object = PickledObjectField(blank=True, null=True)
    engine_object_timestamp = models.DateTimeField(blank=True, null=True)
    network_type = models.SmallIntegerField(choices=NETWORK_TYPE_CHOICES,
                                            default=TYPE_GENERAL,
                                            blank=True, null=True)
    metadata = JSONField(default={}, blank=True, null=True)
    engine_meta_iterations = models.SmallIntegerField(default=1)
    engine_iterations = models.SmallIntegerField(default=1000)
    results_storage = models.CharField("Results Storage", max_length=100,
                                       blank=True, null=True)
    counter = models.IntegerField(default=0, blank=True, null=True)
    counter_threshold = models.IntegerField(blank=True, null=True)
    threshold_actions = models.CharField(max_length=200,
                                         blank=True, null=True)

    def __str__(self):
        return("[BN: {0}]".format(self.name))

    def save(self, *args, **kwargs):
        # Initialize metadata field if corresponds
        if self.metadata == {}:
            if self.network_type == self.TYPE_CLUSTERING:
                self.metadata["clusters_labels"] = {}
                self.metadata["prev_clusters_labels"] = {}
                self.metadata["clusters_means"] = {}
                self.metadata["prev_clusters_means"] = {}
                self.metadata["clusters_sizes"] = {}
                self.metadata["prev_clusters_sizes"] = {}
                self.metadata["columns"] = []
        # Runs threshold actions if corresponds
        self.parse_and_run_threshold_actions()

        super(BayesianNetwork, self).save(*args, **kwargs)

    def clean(self):
        if self.results_storage:
            # Check the validity of results_storage field
            try:
                rs = self.parse_results_storage()
            except Exception as e:
                msg = e.args[0]
                raise ValidationError({'results_storage': _(
                    'Invalid format or storage engine: {}'.format(msg)
                )})
            if rs["storage"] == "dmf":
                try:
                    model_class = ContentType.objects.get(
                        app_label=rs["attrs"]["app"],
                        model=rs["attrs"]["model"].lower()
                    ).model_class()
                except Exception as e:
                    msg = e.args[0]
                    raise ValidationError({'results_storage': _(
                        'Error getting the model: {}'.format(msg)
                    )})
                try:
                    getattr(model_class, rs["attrs"]["field"])
                except Exception as e:
                    msg = e.args[0]
                    raise ValidationError({'results_storage': _(
                        'Error accessing the field: {}'.format(msg)
                    )})
        # Check threshold_actions keywords are valid
        if self.threshold_actions:
            for action in self.threshold_actions.split(" "):
                if not action in self.ACTIONS_KEYWORDS:
                    raise ValidationError({'threshold_actions': _(
                        'Unrecognized action: {}'.format(action)
                    )})

    def parse_and_run_threshold_actions(self):
        if self.counter_threshold:
            if self.counter_threshold <= self.counter:
                self.counter = 0
                actions = self.threshold_actions.split(" ")
                for action in actions:
                    if action == ":recalculate":
                        self.perform_inference(recalculate=True)
                return(True)
        else:
            return(False)

    def parse_results_storage(self):
        storage, attrs = self.results_storage.split(":", 1)
        if storage == "dmf":
            app, model, field = attrs.split(".")
            return(
                {
                    "storage": storage,
                    "attrs": {
                        "app": app,
                        "model": model,
                        "field": field
                        }
                }
            )
        else:
            raise ValueError(_(
                '"{}" engine is not implemented.'.format(storage)
            ))

    def get_graph(self):
        dot = Digraph(comment=self.name)
        nodes = self.nodes.all()
        for node in nodes:
            dot.node(name=node.name, label=node.name)
        edges = self.edges.all()
        for edge in edges:
            # Pending: debug constraint param
            dot.edge(str(edge.parent.name),
                     str(edge.child.name))
        return(dot)

    def get_raw_network_image(self):
        dot = self.get_graph()
        dot.format = "png"
        return ContentFile(dot.pipe())

    def update_image(self):
        if self.image:
            self.image.delete()
        image_name = "{0}/{1}".format(
            os.path.join("django_ai",
                         "bayesian_networks"),
            self.name + ".png")
        self.image.save(image_name,
                        self.get_raw_network_image())

    def get_nodes_names(self):
        return(self.nodes.all().values_list("name", flat=True))

    def get_engine_object(self, reconstruct=False, propagate=True, save=True):
        """
        Constructs the Engine Objects of all Nodes in the Network for
        initializing the Inference Engine of the Network.

        This is the method that should be called for initializing the EOs of
        the Nodes, as it handle the dependencies correctly and then propagate
        them to the Nodes' objects.

        CAVEAT: You might have to call 'node.refresh_from_db()' if for some
        reason the Nodes are already retrieved before this method is ran.
        """
        if self.engine_object and not reconstruct:
            return(self.engine_object)

        nodes = self.nodes.all()
        # Initialize the "EOs Struct" - a structure needed for keeping the
        # same Python Objects in order to correctly initialize the BP's
        # Inference Engine.
        # {node_name: {django_model:, engine_object:}}
        eos_struct = {n.name: {"dm": n, "eo": None}
                      for n in nodes}

        # Root nodes have no problem getting their engine object and serve
        # as the recursion base step
        root_nodes = {n.name:
                      {"dm": n,
                       "eo": n.get_engine_object(reconstruct=reconstruct)
                       }
                      for n in nodes if not n.parents()}
        # Update with the root nodes
        eos_struct.update(root_nodes)

        # Child nodes are "intermediate" nodes and "final" ones ("leafs" in
        # trees). These are the ones that need the EOs Struct for recursion
        child_nodes = [n for n in nodes if n.parents()]
        for cn in child_nodes:
            eos_struct = BayesianNetwork.update_eos_struct(eos_struct, cn)

        # Collect the EOs
        nodes_eos = []
        for node in eos_struct:
            nodes_eos.append(eos_struct[node]["eo"])

        # Initialize the BP's Inference Engine
        self.engine_object = bp.inference.VB(*nodes_eos)

        # Propagate to the network
        if propagate:
            for node in eos_struct:
                eos_struct[node]["dm"].engine_object = self.engine_object[node]

        # Save the BN and its nodes
        if save:
            self.engine_object_timestamp = timezone.now()
            self.save()
            if propagate:
                for node in eos_struct:
                    eos_struct[node]["dm"].save()

        return(self.engine_object)

    def perform_inference(self, iters=None, recalculate=False, save=True):
        """
        Retrieves the Engine Object of the Network, performs the inference
        and propagates the results to the Nodes.
        """
        if not self.engine_object or recalculate:
            # If iters is not set, use the model's which defaults to 100
            if not iters:
                iters = self.engine_iterations
            # Run the inference 'e_m_i' times
            for i in range(self.engine_meta_iterations):
                Q = self.get_engine_object(reconstruct=True)
                # Run the inference
                Q.update(repeat=iters)
                # Save the result in the internal variable
                self._eo_meta_iterations[i] = {"eo": Q, "L": max(Q.L)}
            # Use the eo with the highest likelihood
            hl_eo = max(self._eo_meta_iterations,
                        key=lambda x: self._eo_meta_iterations[x]["L"])
            self.engine_object = self._eo_meta_iterations[hl_eo]["eo"]
            if save:
                self.engine_object_timestamp = timezone.now()
                self.save()
                # Propagate results to the network
                for node in self.nodes.all():
                    node.engine_inferred_object = \
                        self.engine_object[node.name]
                    node.engine_object_timestamp = timezone.now()
                    node.update_image()
                    node.save()
            # Update metadata
            if self.network_type == self.TYPE_CLUSTERING:
                self.assign_clusters_labels(save=save)
                self.columns_names_to_metadata(save=save)
                self.metadata_update_cluster_sizes(save=save)
                self.store_results()
        return(self.engine_object)

    def reset_engine_object(self, save=True):
        """
        Resets the Engine Object and timestamp
        """
        self.engine_object = None
        self.engine_object_timestamp = None
        self.metadata = {}
        if save:
            self.save()
        return(True)

    def reset_inference(self, save=True):
        """
        Resets the Engine Object and timestamp from the Network
        (the Network object itself and all the Nodes objects in it)
        """
        self.reset_engine_object(save=save)
        for node in self.nodes.all():
            node.reset_inference(save=save)
        if self.network_type == self.TYPE_CLUSTERING:
            self.store_results(reset=True)
        return(True)

    def assign_cluster(self, observation):
        """
        If the Network is for Clustering, returns the cluster label for a
        new observation given the current state of the inference.
        Assumptions:
            - The network has a topology of a Gaussian Mixture Model with
              one Categorical Node for cluster assigments with prior Dirichlet
              probabilities
        """
        if self.network_type == self.TYPE_CLUSTERING:
            if not self.is_inferred:
                return(False)
            # if not self._clusters_labels:
            #     self.assign_clusters_labels()
            eo = self.engine_object
            prior_clusters_probs = self.nodes.get(
                distribution=bp_consts.DIST_DIRICHLET).engine_inferred_object
            clusters_means = self.nodes.get(
                distribution=bp_consts.DIST_GAUSSIAN).engine_inferred_object
            clusters_cov_matrices = self.nodes.get(
                distribution=bp_consts.DIST_WISHART).engine_inferred_object
            #
            Z_new = bp.nodes.Categorical(prior_clusters_probs)
            Z_new.initialize_from_random()
            Y_new = bp.nodes.Mixture(Z_new, bp.nodes.Gaussian,
                                     clusters_means, clusters_cov_matrices)
            Y_new.observe(observation)
            Q_0 = bp.inference.VB(Z_new, Y_new, *eo.model)
            Q_0.update(Z_new)
            cluster_label = self.metadata["clusters_labels"][
                str(np.argmax(Z_new.get_moments()[0]))]
            return(cluster_label)
        else:
            return(False)

    def assign_clusters_labels(self, save=True):
        """
        Filters the clusters and constructs a dict for labelling
        Assumptions:
            - The network as a topology of a Gaussian Mixture Model
        """
        clusters_means_node_eo = self.nodes.get(
            distribution=bp_consts.DIST_GAUSSIAN).engine_inferred_object
        clusters_means = clusters_means_node_eo.get_moments()[0]
        cmean_dim = len(clusters_means[0])
        origin = np.zeros(cmean_dim)
        filtered_means = list(np.unique(clusters_means, axis=0))
        # Sort cluster means by norm
        filtered_means.sort(key=np.linalg.norm)
        if not self.metadata["clusters_labels"] == {}:
            self.metadata["prev_clusters_labels"] = self.metadata[
                "clusters_labels"]
            self.metadata["clusters_labels"] = {}
        if not self.metadata["clusters_means"] == {}:
            self.metadata["prev_clusters_means"] = self.metadata[
                "clusters_means"]
            self.metadata["clusters_means"] = {}
        for index, cm in enumerate(clusters_means):
            fm_index = np.argwhere(filtered_means == cm)[0][0]
            cluster_label = self._alphabet[fm_index]
            self.metadata["clusters_labels"][str(index)] = cluster_label
            self.metadata["clusters_means"][cluster_label] = cm
        if save:
            self.save()

    def columns_names_to_metadata(self, save=True):
        """
        Gets the columns names and stores them in metadata
        Assumptions:
            - The network as a topology of a Gaussian Mixture Model
        """
        observations_node = self.nodes.get(
            distribution=bp_consts.DIST_MIXTURE)
        columns = observations_node.columns.order_by("position")
        columns_names = columns.values_list("ref_column", flat=True)
        self.metadata["columns"] = list(columns_names)
        if save:
            self.save()

    def metadata_update_cluster_sizes(self, save=True):
        results = self.get_results()
        # Move cluster_sizes to prev_c_s if it already existed
        if not self.metadata["clusters_sizes"] == {}:
            self.metadata["prev_clusters_sizes"] = self.metadata[
                "clusters_sizes"]
        sizes = dict(Counter(results))
        # Add empty clusters
        for label in self.metadata["clusters_labels"].values():
            if not label in sizes:
                sizes[label] = 0
        self.metadata["clusters_sizes"] = sizes 
        if save:
            self.save()

    def get_results(self):
        """
        Returns a list with the clustering assignation for each observations.
        Assumptions:
                - The network as a topology of a Gaussian Mixture Model
        """
        if self.network_type == self.TYPE_CLUSTERING:
            if not self.is_inferred:
                return(False)
            clusters_assig_eo = self.nodes.get(
                distribution=bp_consts.DIST_CATEGORICAL).engine_inferred_object
            assignations = []
            for assig in clusters_assig_eo.get_moments()[0]:
                label = self.metadata["clusters_labels"][str(np.argmax(assig))]
                assignations.append(label)
            return(assignations)
        else:
            return(None)

    def store_results(self, reset=False):
        """
        Stores the results of the inference of a network in a Model's field
        (to be generalized for other storage options).

        Note that it will update the results using the default ordering of the
        Model in which will be stored.
        """
        if self.results_storage:
            results = self.get_results()
            # results_storage already validated
            rs = self.parse_results_storage()
            if rs["storage"] == "dmf":
                app, model, field = (rs["attrs"]["app"], rs["attrs"]["model"],
                                     rs["attrs"]["field"])
                model_class = ContentType.objects.get(
                    app_label=app,
                    model=model.lower()
                ).model_class()
                if reset:
                    model_class.objects.all().update(**{field: None})
                else:
                    # Prevent from new records
                    model_objects = model_class.objects.all()[:len(results)]
                    # This could be done with django-bulk-update
                    # but for not adding another dependency:
                    for index, model_object in enumerate(model_objects):
                        setattr(model_object, field, results[index])
                        model_object.save()
            return(True)
        else:
            return(False)

    @staticmethod
    def update_eos_struct(eos_struct, node):
        """
        Auxiliary recursive function to "populate" a "branch" (all the ancestors
        of the node) of the "EOs Struct" of the Network DAG.

        By how BayesPy is implemented, this is needed to hold the same objects
        in order to correctly initialize the BP inference engine ("simple
        recursion" via Django models does not work, different "branches" may point
        to different instances of objects, as they are initialized independently)
        """
        parents = [p.name for p in node.parents()]
        # Check if the parents' eos are already present, otherwise initialize them
        # recursively (until there )
        for parent in parents:
            if not eos_struct[parent]["eo"]:
                BayesianNetwork.update_eos_struct(
                    eos_struct, eos_struct[parent]["dm"])
        # Initialize the Node EO and store it in the matrix
        eos_struct[node.name]["eo"] = \
            node.get_engine_object(parents=eos_struct, reconstruct=True)
        return(eos_struct)

    @property
    def is_inferred(self):
        return(self.engine_object is not None)



class BayesianNetworkNodeColumn(models.Model):
    """
    A dimension / axis / column of an Observable Bayesian Network Node.
    """
    node = models.ForeignKey("BayesianNetworkNode",
                             on_delete=models.CASCADE, related_name="columns")
    ref_model = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    ref_column = models.CharField("Reference Column", max_length=100)
    position = models.SmallIntegerField(blank=True, null=True)

    class Meta:
        verbose_name = "Bayesian Networks Node Column"
        verbose_name_plural = "Bayesian Networks Node Columns"
        unique_together = [("node", "ref_model", "ref_column"),
                           ("node", "position")]

    def __str__(self):
        return("{0} | {1} - {2}".format(self.node, self.ref_model,
                                        self.ref_column))

    def clean(self):
        # Check the validity of the Reference Column
        try:
            mc = self.ref_model.model_class()
        except Exception as e:
            raise ValidationError({'ref_model': _(
                'The Reference Model must be a valid Django Model'
            )})
        try:
            getattr(mc, self.ref_column)
        except Exception as e:
            raise ValidationError({'ref_column': _(
                'The column must be a valid attribute of '
                'the ' + self.ref_model.name + ' model'
            )})


class BayesianNetworkNode(models.Model):
    """
    A Node in a Bayesian Network
    """
    _engine_object = None  # Deprecated?
    _engine_inferred_object = None
    _data = None

    NODE_TYPE_STOCHASTIC = 0
    NODE_TYPE_DETERMINISTIC = 1
    NODE_TYPE_CHOICES = (
        (NODE_TYPE_STOCHASTIC, "Stochastic"),
        (NODE_TYPE_DETERMINISTIC, "Deterministic"),
    )
    FIELDS_STOCHASTIC = [
        "is_observable", "distribution", "distribution_params",
        "graph_interval"
    ]
    FIELDS_DETERMINISTIC = [
        "deterministic", "deterministic_params"
    ]

    network = models.ForeignKey(BayesianNetwork, related_name="nodes")
    name = models.CharField("Node Name", max_length=50)
    node_type = models.SmallIntegerField("Type", choices=NODE_TYPE_CHOICES,
                                         default=NODE_TYPE_STOCHASTIC)
    is_observable = models.BooleanField("Is Observable?", default=False)
    distribution = models.CharField("Distribution", max_length=50,
                                    choices=bp_consts.DISTRIBUTION_CHOICES,
                                    blank=True, null=True)
    distribution_params = models.CharField("Distribution Parameters",
                                           max_length=200,
                                           blank=True, null=True)
    deterministic = models.CharField("Deterministic", max_length=50,
                                     choices=bp_consts.DETERMINISTIC_CHOICES,
                                     blank=True, null=True)
    deterministic_params = models.CharField("Deterministic Parameters",
                                            max_length=200,
                                            blank=True, null=True)
    engine_object = PickledObjectField(blank=True, null=True)
    engine_object_timestamp = models.DateTimeField(blank=True, null=True)
    engine_inferred_object = PickledObjectField(blank=True, null=True)
    engine_inferred_object_timestamp = models.DateTimeField(
        blank=True, null=True)
    graph_interval = models.CharField("Graph Interval", max_length=20,
                                      blank=True, null=True)
    image = models.ImageField("Image", blank=True, null=True)

    def parents(self):
        """
        Returns the Parents of the Node as a QuerySet
        """
        parents_ids = self.child_edges.values_list("parent", flat=True)
        return(BayesianNetworkNode.objects.filter(id__in=parents_ids))

    def children(self):
        """
        Returns the Children of the Node as a QuerySet
        """
        children_ids = self.parent_edges.values_list("child", flat=True)
        return(BayesianNetworkNode.objects.filter(id__in=children_ids))

    def clean(self):
        error_dict = {}
        # FIRST STEP: Check there are no fields that don't correspond to
        # the Node type
        if self.node_type == self.NODE_TYPE_STOCHASTIC:
            for field in self.FIELDS_DETERMINISTIC:
                if getattr(self, field) is not None:
                    error_dict[field] = _(
                        'Using Deterministic fields on a Stochastic '
                        'Node is not allowed')
        else:
            for field in self.FIELDS_STOCHASTIC:
                field_value = getattr(self, field)
                # For avoiding the use of NullBooleanField in 'is_observable'
                if field_value is False:
                    field_value = None
                if field_value is not None:
                    error_dict[field] = _(
                        'Using Stochastic fields on a Deterministic '
                        'Node is not allowed')
        if error_dict:
            raise ValidationError(error_dict)
        # SECOND STEP: Check validity on Stochastic Nodes
        if self.node_type == self.NODE_TYPE_STOCHASTIC:
            # Validations on the Distribution of Stochastic Nodes
            if self.distribution is None:
                error_dict['distribution'] = _(
                    'A Stochastic Node must have a distribution')
            if self.distribution_params is None:
                error_dict['distribution_params'] = _(
                    'A Stochastic Node must have a distribution'
                    'parameters')
            # Raise if any
            if error_dict:
                raise ValidationError(error_dict)
        # THIRD STEP: Check Validity on Deterministic Nodes
        else:
            if self.deterministic is None:
                error_dict['deterministic'] = _(
                    'A Deterministic Node must have a function')
            if self.deterministic_params is None:
                error_dict['deterministic_params'] = _(
                    'A Deterministic Node must have deterministic'
                    'parameters')
            # Raise if any
            if error_dict:
                raise ValidationError(error_dict)

        # FOURTH STEP: Check args parsing
        try:
            params = parse_node_args(self.get_params(), flat=True)
        except Exception as e:
            msg = e.args[0]
            raise ValidationError({
                self.get_params_type() + "_params": msg
            })

        # FINAL STEP: Check if the Engine Object (BayesPy) can be initialized
        # Check only if the Node hasn't other Nodes as params (otherwise the
        # networkd Edges should have been created already to resolve the names
        # to Nodes)

        if not any([isinstance(p, str) for p in params]):
            try:
                self.get_engine_object(reconstruct=True, save=False)
            except Exception as e:
                msg = e.args[0]
                raise ValidationError({
                    "distribution_params": "[BayesPy] " + msg})

    def get_data(self):
        """
        Returns a list of R^d points, represented as list of length d,
        constructed from the Node's columns.
        """
        if not self.is_observable:
            return(False)
        data = {}
        columns = self.columns.all().order_by("position")
        if len(columns) == 0:
            raise ValueError(_("No columns defined for an Observable Node"))
        # As they may not be from the same model, the can't be retrieved
        # straight from the ORM
        for column in columns:
            colname = "{0}.{1}".format(column.ref_model, column.ref_column)
            data[colname] = column.ref_model.model_class().objects.values_list(
                column.ref_column, flat=True)
        # and the len of the columns shouls be checked
        lengths = [len(col) for col in data]
        h = lengths[0]
        if any([h == t for t in lengths[1:]]):
            raise ValidationError(
                {"ref_column": _("Columns lengths does not match.")})
        # Construct the list
        data_list = np.stack([data[col] for col in data], axis=-1)
        return(data_list)

    def get_params(self):
        """
        Returns the params according to the node type
        """
        if self.node_type == self.NODE_TYPE_STOCHASTIC:
            return(self.distribution_params)
        else:
            return(self.deterministic_params)

    def get_params_type(self):
        """
        Returns the node type as a string
        """
        if self.node_type == self.NODE_TYPE_STOCHASTIC:
            return("distribution")
        else:
            return("deterministic")

    def get_parents_names(self):
        return(self.parents.objects.values_list("name", flat=True))

    def parse_nodes_in_params(self, params_str):
        """
        Returns the nodes names passed as params to the node
        """
        nodes_in_bn = self.network.get_nodes_names()
        p_params = parse_node_args(params_str, flat=True)
        nodes_in_params = [n for n in p_params if n in nodes_in_bn]
        return(nodes_in_params)

    def resolve_eos_in_params(self, params=[], kwparams={}, parents={}):
        """
        Resolve Nodes' EOs from names passed in params and kwparams
        """
        error_str = _("Can't resolve name to node"
                      " - maybe Network Edges are missing?")
        for index, param in enumerate(params):
            if isinstance(param, str) and not param.startswith(":"):
                if param in parents:
                    params[index] = parents[param]["eo"]
                else:
                    # Strings can be only Node names
                    raise ValueError(error_str)
        for kwparam in kwparams:
            if (isinstance(kwparams[kwparam], str) and
                    not kwparams[kwparam].startswith(":")):
                if kwparams[kwparam] in parents:
                    kwparams[kwparam] = parents[kwparam]["eo"]
                else:
                    # Strings can be only Node names
                    raise ValueError(error_str)
        return((params, kwparams))

    def get_engine_object(self, parents={}, reconstruct=False, save=True):
        """
        Method for initializing the Node's Engine Object (currently BayesPy
        only).

        This is meant to be called from the BayesianNetwork object method -
        BN.get_engine_object() - as it handles all the dependecies of the DAG,
        passing the proper parents' objects and then propagating the results
        to the Nodes.

        Otherwise, if not a root node, you will have to provide the parents.
        """
        if self.engine_object and not reconstruct:
            return self.engine_object

        # Parse Params
        parsed_params = parse_node_args(self.get_params())
        params, kwparams = self.resolve_eos_in_params(parsed_params['args'],
                                                      parsed_params['kwargs'],
                                                      parents)
        # Remove Custom Keywords from params
        custom_keywords = []
        for p in params:
            if isinstance(p, str) and p.startswith(":"):
                params.remove(p)
                custom_keywords.append(p)

        kwparams['name'] = self.name

        if self.node_type == self.NODE_TYPE_STOCHASTIC:
            node_distribution = getattr(bp.nodes, self.distribution)
            # Only assign if plates are not provided on observables
            if self.is_observable:
                self._data = self.get_data()
                if 'plates' not in kwparams:
                    kwparams['plates'] = np.shape(self._data)
            else:
                if 'plates' in kwparams:
                    # Process Custom Keywords in plates
                    length = kwparams['plates'][0]
                    if isinstance(length, str) and length.startswith(":dl"):
                        node = self.network.nodes.get(name=length[4:])
                        length = node.get_data().shape[0]
                    if len(kwparams['plates']) == 2:
                        kwparams['plates'] = (length, kwparams['plates'][1])
                    else:
                        kwparams['plates'] = (length, )
            if ":noplates" in custom_keywords:
                del(kwparams['plates'])

            # Initialize the BP Node
            self.engine_object = node_distribution(*params, **kwparams)
            # and if it is observable, load the data
            if self.is_observable:
                data = self.get_data()
                self.engine_object.observe(data)
            # Initializate from random if requested
            if ":ifr" in custom_keywords:
                self.engine_object.initialize_from_random()

        else:
            # Deterministic Type
            node_deterministic_fun = getattr(bp.nodes, self.deterministic)
            # Initialize the BP Node
            self.engine_object = node_deterministic_fun(*params, **kwparams)

        if save:
            self.engine_object_timestamp = timezone.now()
            self.save()
        return(self.engine_object)

    def get_engine_inferred_object(self, recalculate=False, save=True):
        Q = self.network.perform_inference(recalculate=recalculate)
        self.engine_inferred_object = Q[self.name]
        if save:
            self.engine_inferred_object_timestamp = timezone.now()
            self.save()
        return(self.engine_inferred_object)

    def reset_engine_object(self, save=True):
        self.engine_object = None
        self.engine_object_timestamp = None
        if save:
            self.save()
        return(True)

    def reset_inference(self, save=True):
        self.engine_inferred_object = None
        self.engine_inferred_object_timestamp = None
        self.image.delete()
        if save:
            self.save()
        return(True)

    def update_image(self):
        if (not self.engine_inferred_object or
                not self.node_type == self.NODE_TYPE_STOCHASTIC):
            return False
        if self.image:
            self.image.delete()

        image_name = "{0}/{1}".format(
            os.path.join("django_ai",
                         "bayesian_networks"),
            self.network.name + "_" + self.name + ".png")

        save = False
        if self.distribution == bp_consts.DIST_GAUSSIAN_ARD:
            if not self.graph_interval:
                return(False)
            a, b = self.graph_interval.split(", ")
            bp.plot.pdf(self.engine_inferred_object,
                        np.linspace(float(a), float(b), num=100),
                        name=self.name)
            bp.plot.pyplot.savefig(settings.MEDIA_ROOT + '/' + image_name)
            bp.plot.pyplot.close()
            save = True
        elif self.distribution == bp_consts.DIST_MIXTURE:
            if self.columns.count() == 2:
                bp.plot.gaussian_mixture_2d(
                    self.engine_inferred_object, scale=2)
                bp.plot.pyplot.savefig(settings.MEDIA_ROOT + '/' + image_name)
                bp.plot.pyplot.close()
                save = True

        if save:
            self.image = image_name
            self.save()

    def __str__(self):
        return(str(self.network) + " - " + self.name)

    class Meta:
        unique_together = ["network", "name", ]


class BayesianNetworkEdge(models.Model):
    """
    An Edge between Bayesian Network Nodes
    """
    parent = models.ForeignKey(BayesianNetworkNode,
                               related_name="parent_edges",
                               on_delete=models.CASCADE)
    child = models.ForeignKey(BayesianNetworkNode,
                              related_name="child_edges",
                              on_delete=models.CASCADE)
    network = models.ForeignKey(BayesianNetwork, related_name="edges")
    description = models.CharField("Description", max_length=50)

    def __str__(self):
        return(self.description)

    def save(self, *args, **kwargs):
        self.network = self.parent.network

        super(BayesianNetworkEdge, self).save(*args, **kwargs)


def update_bn_image(sender, **kwargs):
    kwargs['instance'].network.update_image()


post_save.connect(update_bn_image, sender=BayesianNetworkNode)
post_save.connect(update_bn_image, sender=BayesianNetworkEdge)
