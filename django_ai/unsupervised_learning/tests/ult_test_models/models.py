#
from django.db import models


class UserInfo(models.Model):
    """
    Example User Information Model for testing
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

    class Meta:
        app_label = "ult_test_models"

    def __str__(self):
        return(self)


class UserInfo2(models.Model):
    """
    User Information Model 2 for testing
    """
    avg2 = models.FloatField("Average 2", blank=True, null=True)
    avg_time_pages_b = models.FloatField("Average Time spent on Pages B",
                                         blank=True, null=True)
    cluster_2 = models.CharField("Cluster 1", max_length=1,
                                 blank=True, null=True)

    class Meta:
        app_label = "ult_test_models"

    def __unicode__(self):
        return(self)
