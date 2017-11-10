# -*- coding: utf-8 -*-

from django.db import models


class UserInfo(models.Model):
    """
    Example User Information Model
    """
    SEX_CHOICES = [(0, "M"), (1, "F")]

    age = models.IntegerField("Age")
    sex = models.SmallIntegerField("Sex", choices=SEX_CHOICES,
                                   blank=True, null=True)
    # -> Metrics
    avg1 = models.FloatField("Average 1", blank=True, null=True)
    avg_time_pages = models.FloatField("Average Time on Pages",
                                       default=0, blank=True, null=True)
    visits_pages = models.IntegerField("Visits on Pages (Total)",
                                       default=0)
    avg_time_pages_a = models.FloatField("Average Time on Pages of type A",
                                         default=0, blank=True, null=True)
    visits_pages_a = models.IntegerField("Visits on Pages of type A",
                                         default=0)
    avg_time_pages_b = models.FloatField("Average Time on Pages of type B",
                                         default=0, blank=True, null=True)
    visits_pages_b = models.IntegerField("Visits on Pages of type B",
                                         default=0)
    avg_time_pages_c = models.FloatField("Average Time on Pages of type C",
                                         default=0, blank=True, null=True)
    visits_pages_c = models.IntegerField("Visits on Pages of type C",
                                         default=0)
    avg_time_pages_d = models.FloatField("Average Time on Pages of type D",
                                         default=0, blank=True, null=True)
    visits_pages_d = models.IntegerField("Visits on Pages of type D",
                                         default=0)
    avg_time_pages_e = models.FloatField("Average Time on Pages of type E",
                                         default=0, blank=True, null=True)
    visits_pages_e = models.IntegerField("Visits on Pages of type E",
                                         default=0)
    avg_time_pages_f = models.FloatField("Average Time on Pages of type F",
                                         default=0, blank=True, null=True)
    visits_pages_f = models.IntegerField("Visits on Pages of type F",
                                         default=0)
    avg_time_pages_g = models.FloatField("Average Time on Pages of type G",
                                         default=0, blank=True, null=True)
    visits_pages_g = models.IntegerField("Visits on Pages of type G",
                                         default=0)
    avg_time_pages_h = models.FloatField("Average Time on Pages of type H",
                                         default=0, blank=True, null=True)
    visits_pages_h = models.IntegerField("Visits on Pages of type H",
                                         default=0)
    avg_time_pages_i = models.FloatField("Average Time on Pages of type I",
                                         default=0, blank=True, null=True)
    visits_pages_i = models.IntegerField("Visits on Pages of type I",
                                         default=0)
    avg_time_pages_j = models.FloatField("Average Time on Pages of type J",
                                         default=0, blank=True, null=True)
    visits_pages_j = models.IntegerField("Visits on Pages of type J",
                                         default=0)
    # -> Results
    cluster_1 = models.CharField("Cluster 1", max_length=1,
                                 blank=True, null=True)

    def __unicode__(self):
        return(self)
