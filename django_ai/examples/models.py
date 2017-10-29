# -*- coding: utf-8 -*-

from django.db import models


class UserInfo(models.Model):
    """
    Example User Information Model
    """
    SEX_CHOICES = ((0, "M"), (1, "F"))

    age = models.IntegerField("Age")
    sex = models.CharField("Sex", choices=SEX_CHOICES, max_length=1)
    avg1 = models.FloatField("Average 1", blank=True, null=True)
    avg_time_logged = models.FloatField("Average Weekly Time Logged In",
                                        blank=True, null=True)
    avg_time_pages_a = models.FloatField("Average Time spent on Pages A",
                                         blank=True, null=True)
    cluster_1 = models.CharField("Cluster 1", max_length=1,
                                 blank=True, null=True)

    def __unicode__(self):
        return(self)
