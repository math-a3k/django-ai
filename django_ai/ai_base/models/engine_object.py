from pickle import HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL

from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from picklefield.fields import PickledObjectField


def _get_default_metadata():
    return EngineObjectModel.DEFAULT_METADATA


class EngineObjectModel(models.Model):
    #: Allowed Keywords for Threshold actions
    ACTIONS_KEYWORDS = [":recalculate", ]
    DEFAULT_METADATA = {
        "inference": {
            "current": {}, "previous": {}
        },
        "meta": {"descriptions": {}}
    }

    #: This is where the main object of the Engine resides.
    engine_object = PickledObjectField(
        "Engine Object",
        protocol=pickle_HIGHEST_PROTOCOL,
        blank=True, null=True
    )
    #: The timestamp of the Engine Object creation or last update
    engine_object_timestamp = models.DateTimeField(
        "Engine Object Timestamp",
        blank=True, null=True
    )
    #: If Inference has been performed on the Engine Object
    is_inferred = models.BooleanField(
        "Is Inferred?",
        default=False
    )
    #: Field for storing metadata (results and / or information related to
    #: internal tasks) of the Engine Object
    metadata = models.JSONField(
        "Metadata",
        default=_get_default_metadata, blank=True, null=True
    )
    #: Internal Counter for automating running actions.
    counter = models.IntegerField(
        "Internal Counter",
        default=0, blank=True, null=True,
        help_text=(
            'Internal Counter for automating running actions.'
        )
    )
    #: Automation: Internal Counter Threshold
    counter_threshold = models.IntegerField(
        "Internal Counter Threshold",
        blank=True, null=True,
        help_text=(
            'Threshold of the Internal Counter for triggering the running '
            'of actions.'
        )
    )
    #: Automation: Actions to be run when the threshold is met.
    threshold_actions = models.CharField(
        "Threshold actions",
        max_length=200, blank=True, null=True,
        help_text=(
            'Actions to be run once the Internal Counter has reachd the '
            'Counter Threshold in the ":action_name" format and separated '
            'by a space, i.e. ":recalculate :mail_staff"'
        )
    )

    class Meta:
        abstract = True

    def __str__(self):
        return "[EO] Inferred: {} {}".format(
            self.is_inferred, self.engine_object_timestamp or "-"
        )

    def engine_object_init(self):
        """
        Returns the initialized Engine Object
        """
        raise NotImplementedError("A Technique should implement this method")

    def get_engine_object_conf(self):
        """
        Returns the Engine Object configuration as a dict
        """
        raise NotImplementedError("A Technique should implement this method")

    def engine_object_perform_inference(self, *args, **kwargs):
        """
        Returns the Engine object with the inference performed.
        """
        raise NotImplementedError("A Technique should implement this method")

    def engine_object_inference_scores(self, *args, **kwargs):
        """
        Returns the inference score of the Engine's current state for the
        given data and labels.
        """
        raise NotImplementedError("A Technique should implement this method")

    def engine_object_reset(self, save=True):
        """
        Resets the engine object and related fields.
        (engine_object, engine_object_timestamp, metadata and is_inferred).
        """
        self.engine_object = None
        self.engine_object_timestamp = None
        self.rotate_inference_metadata()
        self.is_inferred = False
        if save:
            self.save()
        return True

    def get_engine_object(self, reconstruct=False):
        """
        Returns the main object provided by the Statistical Engine.
        """
        if self.engine_object is not None and not reconstruct:
            return(self.engine_object)

        self.engine_object = self.engine_object_init()
        return self.engine_object

    def perform_inference(self, save=True):
        """
        Performs the Inference with the Statistical Engine and updates the
        Engine Object and related fields. Engines should define how to perform
        inference in engine_object_perform_inference().
        """
        self.engine_object = self.engine_object_perform_inference()
        self.is_inferred = True
        self.engine_object_timestamp = timezone.now()
        self.rotate_inference_metadata()
        self.metadata["inference"]["current"] = self.get_inference_metadata()
        if save:
            self.save()
        return(self.engine_object)

    def reset_inference(self, save=True):
        """
        Resets the Engine's inference-related fields and rotates metadata
        """
        return self.engine_object_reset(save=save)

    def reset_metadata(self, save=True):
        """
        Resets the Engine's inference-related fields and rotates metadata
        """
        self.metadata = self.DEFAULT_METADATA
        if save:
            self.save()
        return self.metadata

    def rotate_inference_metadata(self):
        """
        Rotates metadata from "current_inference" to "previous_inference" if
        it is not empty.
        """
        if self.metadata["inference"]["current"] != {}:
            self.metadata["inference"]["previous"] = \
                self.metadata["inference"]["current"]
            self.metadata["inference"]["current"] = {}

    def get_inference_metadata(self):
        metadata = {"conf": {"eo": {"params": {}}}}
        metadata["conf"]["eo"]["params"] = self.get_engine_object_conf()
        return metadata

    def run_actions(self, actions=[]):
        actions = actions or [
            a.removeprefix("action_")
            for a in dir(self) if a.startswith("action_")
        ]
        results = {}
        for action in actions:
            technique_action = getattr(self, "action_{}".format(action), None)
            if technique_action:
                results[action] = technique_action()
        return results

    def run_actions_if_threshold(self):
        """
        Runs the actions if corresponds for the state of the
        counter given the counter threshold.
        """
        if self.counter_threshold:
            if self.counter_threshold <= self.counter:
                self.counter = 0
                actions = [a[1:] for a in self.threshold_actions.split(" ")]
                results = self.run_actions(actions)
                return results
        else:
            return False

    def action_recalculate(self):
        self.perform_inference()
        return True

    def _get_metadata_descriptions(self):
        descriptions = {}
        descriptions["conf"] = "Configuration"
        descriptions["eo"] = "Engine"
        descriptions["params"] = "Parameters"
        descriptions["inference"] = "Inference"
        descriptions["previous"] = "Previous"
        descriptions["current"] = "Current"
        descriptions["general"] = "General"
        return descriptions

    # -> Django Models API
    def clean(self):
        # Check threshold_actions keywords are valid
        if self.threshold_actions:
            for action in self.threshold_actions.split(" "):
                if action not in self.ACTIONS_KEYWORDS:
                    raise ValidationError({'threshold_actions': _(
                        'Unrecognized action: {}'.format(action)
                    )})
        super().clean()

    def save(self, *args, **kwargs):
        """
        Base save() processing
        """
        # Ensure structure when resetting metadata
        if not self.metadata.get("inference", None):
            self.metadata["inference"] = {"current": {}, "previous": {}}
        if not self.metadata.get("meta", None):
            self.metadata["meta"] = {}

        # [Re]build metadata descriptions
        self.metadata["meta"]["descriptions"] = self._get_metadata_descriptions()

        # Runs threshold actions if corresponds
        self.run_actions_if_threshold()

        super().save(*args, **kwargs)
