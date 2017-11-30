# -*- coding: utf-8 -*-

from pickle import HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL
import numpy as np

from django.db import models
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import (GenericForeignKey,
                                                GenericRelation, )
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import ValidationError

from picklefield.fields import PickledObjectField
from jsonfield import JSONField


class StatisticalModel(models.Model):
    """
    Metaclass for Learning Techniques.

    It defines the common interface so the Techniques can be "plugged"
    along the framework and the applications.
    """
    SM_TYPE_GENERAL = 0
    SM_TYPE_SUPERVISED = 1
    SM_TYPE_UNSUPERVISED = 2

    SM_TYPE_CHOICES = (
        (SM_TYPE_GENERAL, "General"),
        (SM_TYPE_SUPERVISED, "Classification"),
        (SM_TYPE_UNSUPERVISED, "Regression"),
    )

    ACTIONS_KEYWORDS = []

    name = models.CharField(
        "Name",
        unique=True,
        max_length=100
    )
    engine_object = PickledObjectField(
        "Engine Object",
        protocol=pickle_HIGHEST_PROTOCOL,
        blank=True, null=True
    )
    engine_object_timestamp = models.DateTimeField(
        "Engine Object Timestamp",
        blank=True, null=True
    )
    sm_type = models.SmallIntegerField(
        "Statistical Technique Type",
        choices=SM_TYPE_CHOICES, default=SM_TYPE_GENERAL,
        blank=True, null=True
    )
    metadata = JSONField(
        "Metadata",
        default={}, blank=True, null=True
    )
    engine_meta_iterations = models.SmallIntegerField(
        "Engine Meta Iterations",
        default=1
    )
    engine_iterations = models.SmallIntegerField(
        "Engine Iterations (Max)",
        blank=True, null=True
    )
    results_storage = models.CharField(
        "Results Storage",
        max_length=100, blank=True, null=True
    )
    counter = models.IntegerField(
        "Internal Counter",
        default=0, blank=True, null=True
    )
    counter_threshold = models.IntegerField(
        "Internal Counter Threshold",
        blank=True, null=True
    )
    threshold_actions = models.CharField(
        "Threshold actions",
        max_length=200, blank=True, null=True
    )
    data_columns = GenericRelation(
        "base.DataColumn",
        related_query_name="%(app_label)s_%(class)ss",
    )

    class Meta:
        abstract = True
        verbose_name = "Statistical Technique"
        verbose_name_plural = "Statistical Techniques"

    def __str__(self):
        return("[ST|{0}]".format(self.name))

    # -> Public API
    def get_engine_object(self):
        raise NotImplementedError("A Technique should implement this method")

    def reset_engine_object(self):
        raise NotImplementedError("A Technique should implement this method")

    def perform_inference(self):
        raise NotImplementedError("A Technique should implement this method")

    def reset_inference(self):
        raise NotImplementedError("A Technique should implement this method")

    def get_data(self):
        """
        Returns a list of R^d points, represented as list of length d,
        constructed from the Node's columns.
        """
        data = {}
        columns = self.data_columns.all().order_by("position")
        if len(columns) == 0:
            raise ValueError(_("No columns defined for the Model / Technique"))
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

    def store_results(self, reset=False):
        """
        Stores the results of the inference of a network in a Model's field
        (to be generalized for other storage options).

        Note that it will update the results using the default ordering of the
        Model in which will be stored.
        """
        if self.sm_type != self.SM_TYPE_GENERAL and self.results_storage:
            self._store_results(reset=reset)
            return(True)
        else:
            return(False)

    # -> Django Models API
    def clean(self):
        if self.results_storage:
            # Check the validity of results_storage field
            try:
                rs = self._parse_results_storage()
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
                if action not in self.ACTIONS_KEYWORDS:
                    raise ValidationError({'threshold_actions': _(
                        'Unrecognized action: {}'.format(action)
                    )})

    # -> Internal API
    def _parse_results_storage(self):
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

    def _store_results(self, reset=False):
        results = self.get_results()
        # results_storage already validated
        rs = self._parse_results_storage()
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


class SupervisedLearningTechnique(StatisticalModel):
    """
    Metaclass for Supervised Learning Techniques.
    """
    SL_TYPE_CLASSIFICATION = 0
    SL_TYPE_REGRESSION = 1

    SL_TYPE_CHOICES = (
        (SL_TYPE_CLASSIFICATION, "Classification"),
        (SL_TYPE_REGRESSION, "Regression"),
    )

    sl_type = models.SmallIntegerField(
        "Supervised Learning Type",
        choices=SL_TYPE_CHOICES, default=SL_TYPE_CLASSIFICATION,
        blank=True, null=True
    )
    labels_column = models.CharField(
        "Labels' Column",
        max_length=100, blank=True, null=True,
        help_text=_((
            'Format: app_label.model.attribute'
        ))
    )

    class Meta:
        abstract = True
        verbose_name = "Supervised Learning Technique"
        verbose_name_plural = "Supervised Learning Techniques"
        app_label = "supervised_learning"

    def __init__(self, *args, **kwargs):
        kwargs["sm_type"] = self.SM_TYPE_SUPERVISED
        super(SupervisedLearningTechnique, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[SL|{0}]".format(self.name))

    # -> Public API
    def predict(self, sl_input):
        raise NotImplementedError("A Technique should implement this method")

    def get_labels(self):
        app, model, attribute = self.labels_column.split(".")
        model_class = ContentType.objects.get(
            app_label=app,
            model=model.lower()
        ).model_class()
        labels = model_class.objects.values_list(attribute, flat=True)
        return(labels)

    # -> Django Models API
    def clean(self):
        if self.labels_column:
            # Check the validity of the Labels Column
            try:
                app, model, attribute = self.labels_column.split(".")
            except Exception:
                raise ValidationError({'labels_column': _(
                    'Invalid format'
                )})
            try:
                model_class = ContentType.objects.get(
                    app_label=app,
                    model=model.lower()
                ).model_class()
            except Exception:
                raise ValidationError({'labels_column': _(
                    'The Reference Model must be a valid Django Model'
                )})
            try:
                getattr(model_class, attribute)
            except Exception:
                raise ValidationError({'labels_column': _(
                    'The column must be a valid attribute of '
                    'the {} model'.format(model_class._meta.verbose_name)
                )})


class UnsupervisedLearningTechnique(StatisticalModel):
    """
    Metaclass for Supervised Learning Techniques.
    """
    UL_TYPE_CLUSTERING = 0
    UL_TYPE_OTHER = 1

    UL_TYPE_CHOICES = (
        (UL_TYPE_CLUSTERING, "Clustering"),
        (UL_TYPE_OTHER, "Other"),
    )

    ul_type = models.SmallIntegerField(
        "Unsupervised Learning Type",
        choices=UL_TYPE_CHOICES, default=UL_TYPE_CLUSTERING,
        blank=True, null=True
    )

    class Meta:
        abstract = True
        verbose_name = "Unsupervised Learning Technique"
        verbose_name_plural = "Unsupervised Learning Techniques"

    def __init__(self, *args, **kwargs):
        kwargs["sm_type"] = self.SM_TYPE_UNSUPERVISED
        super(UnsupervisedLearningTechnique, self).__init__(*args, **kwargs)

    def __str__(self):
        return("[UL|{0}]".format(self.name))

    # -> Public API
    def assign(self, sl_input):
        raise NotImplementedError("A Technique should implement this method")

    def get_results(self):
        raise NotImplementedError("A Technique should implement this method")


class DataColumn(models.Model):
    """
    A dimension / axis / column of Technique / Model.
    """
    # -> Model or Technique reference
    content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE
    )
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey(
        'content_type', 'object_id'
    )
    # -> Data reference
    ref_model = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        related_name="%(app_label)s_%(class)ss"
    )
    ref_column = models.CharField(
        "Reference Column",
        max_length=100
    )
    position = models.SmallIntegerField(
        "Position",
        blank=True, null=True
    )

    class Meta:
        verbose_name = "Data Column"
        verbose_name_plural = "Data Columns"
        unique_together = [("content_type", "object_id",
                            "ref_model", "ref_column"),
                           ("content_type", "object_id", "position")]
        app_label = "base"

    def __str__(self):
        return("{0} | {1} - {2}".format(self.content_type, self.ref_model,
                                        self.ref_column))

    def clean(self):
        # Check the validity of the Reference Column
        try:
            mc = self.ref_model.model_class()
        except Exception:
            raise ValidationError({'ref_model': _(
                'The Reference Model must be a valid Django Model'
            )})
        try:
            getattr(mc, self.ref_column)
        except Exception:
            raise ValidationError({'ref_column': _(
                'The column must be a valid attribute of '
                'the ' + self.ref_model.name + ' model'
            )})
