from uuid import uuid4

from django.db import models
from django.utils.translation import ugettext_lazy as _

from django_ai.ai_base.models import LearningTechnique


class DataImputer(LearningTechnique):
    technique = models.OneToOneField(
        'ai_base.LearningTechnique',
        related_name='data_imputer_object',
        blank=True, null=True,
        on_delete=models.SET_NULL
    )

    class Meta:
        verbose_name = _("Data Imputer")
        verbose_name_plural = _("Data Imputers")

    def __str__(self):
        return "[Data Imputer] {}".format(super().__str__())

    @classmethod
    def new_imputer_for(cls, technique):
        name = "DI: {}".format(str(uuid4())[:5])
        imputer = cls(name=name, technique=technique)
        imputer.data_model = technique.data_model
        imputer.learning_fields = technique.learning_fields
        imputer.learning_fields_categorical = \
            technique.learning_fields_categorical
        imputer.save()
        imputer.perform_inference()
        return imputer

    def impute_row(self, row):
        raise NotImplementedError("A subclass should implement this method")

    def _get_imputer(self):
        imputers_fields = [
            f.related_query_name() for f in self._meta._relation_tree
            if 'dataimputer_ptr_id' in f.attname
        ]
        if imputers_fields:
            for i_f in imputers_fields:
                if hasattr(self, i_f):
                    return getattr(self, i_f)
        else:
            return self
        return None
