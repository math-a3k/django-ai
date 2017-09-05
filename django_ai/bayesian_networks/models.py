# -*- coding: utf-8 -*-

import os

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

from django_dag.models import node_factory, edge_factory
from picklefield.fields import PickledObjectField

from .bayespy_constants import (DISTRIBUTION_CHOICES,
                                DETERMINISTIC_CHOICES)


class BayesianNetwork(models.Model):
    """
    Model for a Bayesian Network
    """
    _Q = None

    name = models.CharField("Name", max_length=100)
    image = models.ImageField("Image", blank=True, null=True)
    engine_object = PickledObjectField(blank=True, null=True)
    engine_object_timestamp = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return("BN: " + self.name)

    def save(self, *args, **kwargs):
        """
        """
        super(BayesianNetwork, self).save(*args, **kwargs)

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

    def get_engine_object(self, reconstruct=False, save=True):
        if self.engine_object and not reconstruct:
            return(self.engine_object)

        nodes_eos = []
        nodes = self.nodes.all()
        nodes_dict = {n.name: n for n in nodes}
        root_nodes = {n.name: n.get_engine_object(reconstruct=reconstruct)
                      for n in nodes if not n.parents()}
        child_nodes = [n for n in nodes if n.parents()]
        # TODO: Currently support for one level of depth, pending for any graph
        for node_name in root_nodes:
            nodes_eos.append(root_nodes[node_name])
        for c_n in child_nodes:
            p_ns = {n: root_nodes[n] for n in c_n.parse_nodes_in_dist_params()}
            nodes_eos.append(c_n.get_engine_object(
                parents=p_ns, reconstruct=reconstruct))
        self.engine_object = bp.inference.VB(*nodes_eos)

        if save:
            self.engine_object_timestamp = timezone.now()
            self.save()
        return(self.engine_object)

    def perform_inference(self, iters=100, recalculate=False, save=True):
        if not self.engine_object or recalculate:
            Q = self.get_engine_object(reconstruct=True)
            # Run the inference
            Q.update(repeat=iters)
            if save:
                self.engine_object_timestamp = timezone.now()
                self.save()
                # Propagate results to the network
                for node in self.nodes.all():
                    node.engine_inferred_object = Q[node.name]
                    node.engine_object_timestamp = timezone.now()
                    node.update_image()
                    node.save()
        return(self.engine_object)

    def get_inf_object(self):
        return(self.engine_object)

    def reset_engine_object(self, save=True):
        self.engine_object = None
        self.engine_object_timestamp = None
        if save:
            self.save()
        return(True)

    def reset_inference(self, save=True):
        self.reset_engine_object(save=save)
        for node in self.nodes.all():
            node.reset_inference(save=save)
        return(True)

    @property
    def is_inferred(self):
        return(self.engine_object is not None)


class BayesianNetworkEdge(
        edge_factory('bayesian_networks.BayesianNetworkNode')):
    """
    An Edge between Bayesian Network Nodes
    """
    network = models.ForeignKey(BayesianNetwork, related_name="edges")
    description = models.CharField("Description", max_length=50)

    def __str__(self):
        return(self.description)

    def save(self, *args, **kwargs):
        self.network = self.parent.network
        super(BayesianNetworkEdge, self).save(*args, **kwargs)


class BayesianNetworkNode(
        node_factory('bayesian_networks.BayesianNetworkEdge')):
    """
    A Node in a Bayesian Network
    """
    _engine_object = None  # Deprecated?
    _engine_inferred_object = None

    NODE_TYPE_STOCHASTIC = 0
    NODE_TYPE_DETERMINISTIC = 1
    NODE_TYPE_CHOICES = (
        (NODE_TYPE_STOCHASTIC, "Stochastic"),
        (NODE_TYPE_DETERMINISTIC, "Deterministic"),
    )

    network = models.ForeignKey(BayesianNetwork, related_name="nodes")
    name = models.CharField("Node Name", max_length=50)
    node_type = models.SmallIntegerField("Type", choices=NODE_TYPE_CHOICES,
                                         default=NODE_TYPE_STOCHASTIC)
    is_observable = models.BooleanField("Is Observable?", default=True)
    distribution = models.CharField("Distribution", max_length=50,
                                    choices=DISTRIBUTION_CHOICES,
                                    blank=True, null=True)
    distribution_params = models.CharField("Distribution Parameters",
                                           max_length=200,
                                           blank=True, null=True)
    deterministic = models.CharField("Deterministic", max_length=50,
                                     choices=DETERMINISTIC_CHOICES,
                                     blank=True, null=True)
    deterministic_params = models.CharField("Deterministic Parameters",
                                            max_length=200,
                                            blank=True, null=True)
    ref_model = models.ForeignKey(ContentType, on_delete=models.CASCADE,
                                  blank=True, null=True)
    ref_column = models.CharField("Reference Column", max_length=100,
                                  blank=True, null=True)
    engine_object = PickledObjectField(blank=True, null=True)
    engine_object_timestamp = models.DateTimeField(blank=True, null=True)
    engine_inferred_object = PickledObjectField(blank=True, null=True)
    engine_inferred_object_timestamp = models.DateTimeField(
        blank=True, null=True)
    graph_interval = models.CharField("Graph Interval", max_length=20,
                                      blank=True, null=True)
    image = models.ImageField("Image", blank=True, null=True)

    def get_data(self):
        if not self.is_observable:
            return(False)
        data = self.ref_model.model_class().objects.values_list(
            self.ref_column, flat=True)
        return(data)

    def get_parents_names(self):
        return(self.parents.objects.values_list("name", flat=True))

    def parse_nodes_in_dist_params(self):
        """
        Returns the nodes names passed as params to the node distribution
        """
        nodes_in_bn = self.network.get_nodes_names()
        nodes_in_params = [n for n in self.distribution_params.split(", ")
                           if n in nodes_in_bn]
        return(nodes_in_params)

    def get_engine_object(self, parents={}, reconstruct=False, save=True):
        # Currently Bayespy only
        if self.engine_object and not reconstruct:
            return self.engine_object
        if self.node_type == self.NODE_TYPE_STOCHASTIC:
            node_distribution = getattr(bp.nodes, self.distribution)
            nodes_in_bn = self.network.get_nodes_names()
            params = []
            kwparams = {'name': self.name}
            # Params currently are only ", "-separated and can be scalars
            # or nodes' names
            for p in self.distribution_params.split(", "):
                if p not in parents:  # nodes_in_bn:
                    params.append(float(p))
                else:
                    # node = BayesianNetworkNode.objects.get(name=p)
                    params.append(parents[p])
            if self.is_observable:
                kwparams['plates'] = (self.ref_model.model_class()
                                      .objects.count(), )
            # import ipdb; ipdb.set_trace()
            self.engine_object = node_distribution(*params, **kwparams)
            # Once initialized, if observable, load the data
            if self.is_observable:
                data = self.get_data()
                self.engine_object.observe(data)
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
                not self.node_type == self.NODE_TYPE_STOCHASTIC or
                not self.graph_interval):
            return False
        if self.image:
            self.image.delete()

        a, b = self.graph_interval.split(", ")
        bp.plot.pdf(self.engine_inferred_object,
                    np.linspace(float(a), float(b), num=100), name=self.name)
        image_name = "{0}/{1}".format(
            os.path.join("django_ai",
                         "bayesian_networks"),
            self.network.name + "_" + self.name + ".png")
        bp.plot.pyplot.savefig(settings.MEDIA_ROOT + '/' + image_name)
        bp.plot.pyplot.close()
        self.image = image_name
        self.save()

    def __str__(self):
        return(str(self.network) + " - " + self.name)

    class Meta:
        unique_together = ["network", "name", ]


def update_bn_image(sender, **kwargs):
    kwargs['instance'].network.update_image()

post_save.connect(update_bn_image, sender=BayesianNetworkNode)
post_save.connect(update_bn_image, sender=BayesianNetworkEdge)
