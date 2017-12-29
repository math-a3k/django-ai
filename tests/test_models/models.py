# -*- coding: utf-8 -*-

from django.db import models

from django_ai.base.models import (
    StatisticalModel,
    SupervisedLearningTechnique,
    UnsupervisedLearningTechnique,
)
from django_ai.systems.spam_filtering.models import (
    IsSpammable,
    SpamFilterPreTraining,
)


class UserInfo(models.Model):
    """
    Example User Information Model
    """
    SEX_CHOICES = ((0, "M"), (1, "F"))

    age = models.IntegerField("Age")
    sex = models.CharField("Sex", choices=SEX_CHOICES, max_length=1)
    avg1 = models.FloatField("Average 1", blank=True, null=True)
    avg_time_pages = models.FloatField("Average Time spent on Pages",
                                       blank=True, null=True)
    avg_time_pages_a = models.FloatField("Average Time spent on Pages A",
                                         blank=True, null=True)
    cluster_1 = models.CharField("Cluster 1", max_length=1,
                                 blank=True, null=True)

    def __unicode__(self):
        return(self)


class UserInfo2(models.Model):
    """
    User Information Model 2
    """
    avg2 = models.FloatField("Average 2", blank=True, null=True)
    avg_time_pages_b = models.FloatField("Average Time spent on Pages B",
                                         blank=True, null=True)
    cluster_2 = models.CharField("Cluster 1", max_length=1,
                                 blank=True, null=True)

    def __unicode__(self):
        return(self)


class MyStatisticalModel(StatisticalModel):
    class Meta:
        app_label = "test_models"


class MySupervisedLearningTechnique(SupervisedLearningTechnique):
    class Meta:
        app_label = "test_models"


class MyUnsupervisedLearningTechnique(UnsupervisedLearningTechnique):
    class Meta:
        app_label = "test_models"


class SpammableModel(IsSpammable):
    SPAM_FILTER = "Spam Filter for tests"
    SPAMMABLE_FIELD = "comment"

    comment = models.TextField("Comment")


class MySFPT(SpamFilterPreTraining):
    pass
