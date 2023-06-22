from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from django_ai.supervised_learning.models.supervised_learning_imputer import (
    SupervisedLearningImputer,
)


class IntrospectionImputer(SupervisedLearningImputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.technique, "LEARNING_PAIR", None):
            if self.technique.sl_type == self.technique.SL_TYPE_CLASSIFICATION:
                self.classification_technique = str(
                    self.technique.__class__
                ).split("'")[1]
                self.regression_technique = self.technique.LEARNING_PAIR
            else:
                self.classification_technique = self.technique.LEARNING_PAIR
                self.regression_technique = str(
                    self.technique.__class__
                ).split("'")[1]
        else:
            raise ValidationError(
                {
                    "data_imputer": _(
                        "Technique {} does not have a LEARNING_PAIR defined for "
                        "Introspection.".format(self.technique)
                    )
                }
            )
