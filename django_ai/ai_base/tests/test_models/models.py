from django.db import models


class UserInfo(models.Model):
    """
    Example User Information Model for testing
    """
    LEARNING_TARGET = 'sex'
    LEARNING_FIELDS = ['avg1', 'cluster_1', 'cluster_2', 'bool_field']
    LEARNING_FIELDS_CATEGORICAL = ['cluster_1', 'cluster_2', 'bool_field']
    LEARNING_FIELDS_MONOTONIC_CONSTRAINTS = "None"

    SEX_CHOICES = ((0, "M"), (1, "F"))
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
        app_label = "test_models"

    def __str__(self):
        return("ui")

    def _get_learning_fields_categorical(self):
        return None

    def _get_learning_fields(self):
        return ['age', 'cluster_1', 'cluster_2', 'bool_field']

    def _get_data_model_attr(self):
        return [1, 2, 3]


class UserInfo2(models.Model):
    """
    User Information Model 2 for testing
    """
    avg2 = models.FloatField("Average 2", blank=True, null=True)
    avg3 = models.FloatField("Average 2", blank=True, null=True)
    avg_time_pages_b = models.FloatField("Average Time spent on Pages B",
                                         blank=True, null=True)
    cluster_2 = models.CharField("Cluster 2", max_length=1,
                                 blank=True, null=True)

    class Meta:
        app_label = "test_models"

    def __str__(self):
        return(self)

    @classmethod
    def _get_class_method(cls):
        return "ClassMethod"
