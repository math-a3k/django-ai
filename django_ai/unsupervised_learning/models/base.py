#
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _

from django_ai.ai_base.models import LearningTechnique


class UnsupervisedLearningTechnique(LearningTechnique):
    """
    Metaclass for Unsupervised Learning Techniques.
    """

    UL_TYPE_CLUSTERING = 0
    UL_TYPE_OTHER = 1

    UL_TYPE_CHOICES = (
        (UL_TYPE_CLUSTERING, "Clustering"),
        (UL_TYPE_OTHER, "Other"),
    )

    ul_type = models.SmallIntegerField(
        "Unsupervised Learning Type",
        choices=UL_TYPE_CHOICES,
        default=UL_TYPE_CLUSTERING,
        blank=True,
        null=True,
    )
    #: Where to store the results (if applicable)
    results_storage = models.CharField(
        "Results Storage", max_length=150, blank=True, null=True
    )

    class Meta:
        verbose_name = "Unsupervised Learning Technique"
        verbose_name_plural = "Unsupervised Learning Techniques"

    def __init__(self, *args, **kwargs):
        kwargs["technique_type"] = self.TYPE_UNSUPERVISED
        super(UnsupervisedLearningTechnique, self).__init__(*args, **kwargs)

    def __str__(self):
        return "[UL] {0}".format(self.name)

    # -> Public API
    def assign(self, input, include_scores):
        raise NotImplementedError("A Technique should implement this method")

    def get_results(self):
        raise NotImplementedError("A Technique should implement this method")

    def store_results(self, reset=False):
        """
        Stores the results of the inference of a Technique in a
        Model's field (to be generalized for other storage options).

        Note that it will update the results using the default ordering of the
        Model in which will be stored.
        """
        if self.results_storage:
            self._store_results(reset=reset)
            return True
        else:
            return False

    # -> Django Models' API
    def clean(self):
        super().clean()
        if self.results_storage:
            # Check the validity of results_storage field
            try:
                rs = self._parse_results_storage()
            except Exception as e:
                msg = e.args[0]
                raise ValidationError(
                    {
                        "results_storage": _(
                            "Invalid format or storage engine: {}".format(msg)
                        )
                    }
                )
            # Currently rs["storage"] == "dmf" only
            try:
                model_class = ContentType.objects.get(
                    app_label=rs["attrs"]["app"],
                    model=rs["attrs"]["model"].lower(),
                ).model_class()
            except Exception as e:
                msg = e.args[0]
                raise ValidationError(
                    {
                        "results_storage": _(
                            "Error getting the model: {}".format(msg)
                        )
                    }
                )
            try:
                getattr(model_class, rs["attrs"]["field"])
            except Exception as e:
                msg = e.args[0]
                raise ValidationError(
                    {
                        "results_storage": _(
                            "Error accessing the field: {}".format(msg)
                        )
                    }
                )

    # -> Internal API
    def _parse_results_storage(self):
        storage, attrs = self.results_storage.split(":", 1)
        if storage == "dmf":
            app, model, field = attrs.split(".")
            return {
                "storage": storage,
                "attrs": {"app": app, "model": model, "field": field},
            }
        else:
            raise ValueError(
                _('"{}" engine is not implemented.'.format(storage))
            )

    def _store_results(self, reset=False):
        results = self.get_results()
        # results_storage already validated
        rs = self._parse_results_storage()
        # Currently rs["storage"] == "dmf" only
        app, model, field = (
            rs["attrs"]["app"],
            rs["attrs"]["model"],
            rs["attrs"]["field"],
        )
        model_class = ContentType.objects.get(
            app_label=app, model=model.lower()
        ).model_class()
        if reset:
            model_class.objects.all().update(**{field: None})
        else:
            # Prevent from new records
            model_objects = model_class.objects.all()[: len(results)]
            # This could be done with django-bulk-update
            # but for not adding another dependency:
            for index, model_object in enumerate(model_objects):
                setattr(model_object, field, results[index])
                model_object.save()
