#
from django.db import models


class UserInfo(models.Model):
    """
    Example User Information Model for testing
    """
    LEARNING_TARGET = 'sex'
    LEARNING_FIELDS = ['avg1', 'cluster_1', 'cluster_2', 'bool_field']
    LEARNING_FIELDS_CATEGORICAL = ['cluster_1', 'cluster_2', 'bool_field']
    LEARNING_FIELDS_MONOTONIC_CONSTRAINTS = "avg1: 0, cluster_1: 0"

    SEX_CHOICES = (("xy", "Male"), ("xx", "Female"))
    CLUSTER_2_CHOICES = ((0, "A"), (1, "B"), (2, "C"))

    age = models.IntegerField("Age")
    sex = models.CharField("Sex", choices=SEX_CHOICES, max_length=1)
    avg1 = models.FloatField("Average 1", blank=True, null=True)
    avg_time_pages = models.FloatField("Average Time spent on Pages",
                                       blank=True, null=True)
    avg_time_pages_a = models.FloatField("Average Time spent on Pages A",
                                         blank=True, null=True)
    cluster_1 = models.CharField("Cluster 1", max_length=1,
                                 blank=True, null=True)
    cluster_2 = models.CharField("Cluster 2", choices=CLUSTER_2_CHOICES, max_length=1,
                                 blank=True, null=True)
    bool_field = models.BooleanField("Boolean Field", null=True)

    class Meta:
        app_label = "sl_test_models"

    def __str__(self):
        return(str(self.avg1))


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
        app_label = "sl_test_models"

    def __str__(self):
        return(str(self.avg2))
