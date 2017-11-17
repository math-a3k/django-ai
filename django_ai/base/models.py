# -*- coding: utf-8 -*-

from pickle import HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL

from django.db import models
from django.contrib.contenttypes.models import ContentType
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
    ST_TYPE_GENERAL = 0
    ST_TYPE_SUPERVISED = 1
    ST_TYPE_UNSUPERVISED = 2

    ST_TYPE_CHOICES = (
        (ST_TYPE_GENERAL, "General"),
        (ST_TYPE_SUPERVISED, "Classification"),
        (ST_TYPE_UNSUPERVISED, "Regression"),
    )

    ACTIONS_KEYWORDS = []

    name = models.CharField("Name", max_length=100)
    engine_object = PickledObjectField(protocol=pickle_HIGHEST_PROTOCOL,
                                       blank=True, null=True)
    engine_object_timestamp = models.DateTimeField(blank=True, null=True)
    st_type = models.SmallIntegerField(choices=ST_TYPE_CHOICES,
                                       default=ST_TYPE_GENERAL,
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

    def get_results(self):
        raise NotImplementedError("A Technique should implement this method")

    def store_results(self, reset=False):
        """
        Stores the results of the inference of a network in a Model's field
        (to be generalized for other storage options).

        Note that it will update the results using the default ordering of the
        Model in which will be stored.
        """
        if self.st_type != self.ST_TYPE_GENERAL and self.results_storage:
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

    sl_type = models.SmallIntegerField(choices=SL_TYPE_CHOICES,
                                       default=SL_TYPE_CLASSIFICATION,
                                       blank=True, null=True)

    class Meta:
        abstract = True
        verbose_name = "Supervised Learning Technique"
        verbose_name_plural = "Supervised Learning Techniques"

    def __init__(self, *args, **kwargs):
        kwargs["st_type"] = self.ST_TYPE_SUPERVISED
        super(SupervisedLearningTechnique, self).__init__()

    def __str__(self):
        return("[SL|{0}]".format(self.name))

    # -> Public API
    def predict(self, sl_input):
        raise NotImplementedError("A Technique should implement this method")


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

    ul_type = models.SmallIntegerField(choices=UL_TYPE_CHOICES,
                                       default=UL_TYPE_CLUSTERING,
                                       blank=True, null=True)

    class Meta:
        abstract = True
        verbose_name = "Unsupervised Learning Technique"
        verbose_name_plural = "Unsupervised Learning Techniques"

    def __init__(self, *args, **kwargs):
        kwargs["st_type"] = self.ST_TYPE_UNSUPERVISED
        super(UnsupervisedLearningTechnique, self).__init__()

    def __str__(self):
        return("[UL|{0}]".format(self.name))

    # -> Public API
    def assign(self, sl_input):
        raise NotImplementedError("A Technique should implement this method")
