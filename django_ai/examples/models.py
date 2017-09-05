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

    def __unicode__(self):
        return(self)
