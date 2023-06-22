# Generated by Django 3.1.5 on 2021-03-01 16:53

from django.db import migrations, models
import django.db.models.deletion
import picklefield.fields


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="LearningTechnique",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "engine_object",
                    picklefield.fields.PickledObjectField(
                        blank=True,
                        editable=False,
                        null=True,
                        protocol=5,
                        verbose_name="Engine Object",
                    ),
                ),
                (
                    "engine_object_timestamp",
                    models.DateTimeField(
                        blank=True,
                        null=True,
                        verbose_name="Engine Object Timestamp",
                    ),
                ),
                (
                    "is_inferred",
                    models.BooleanField(
                        default=False, verbose_name="Is Inferred?"
                    ),
                ),
                (
                    "metadata",
                    models.JSONField(
                        blank=True,
                        default=dict,
                        null=True,
                        verbose_name="Metadata",
                    ),
                ),
                (
                    "counter",
                    models.IntegerField(
                        blank=True,
                        default=0,
                        help_text="Internal Counter for automating running actions.",
                        null=True,
                        verbose_name="Internal Counter",
                    ),
                ),
                (
                    "counter_threshold",
                    models.IntegerField(
                        blank=True,
                        help_text="Threshold of the Internal Counter for triggering the running of actions.",
                        null=True,
                        verbose_name="Internal Counter Threshold",
                    ),
                ),
                (
                    "threshold_actions",
                    models.CharField(
                        blank=True,
                        help_text='Actions to be run once the Internal Counter has reachd the Counter Threshold in the ":action_name" format and separated by a space, i.e. ":recalculate :mail_staff"',
                        max_length=200,
                        null=True,
                        verbose_name="Threshold actions",
                    ),
                ),
                (
                    "name",
                    models.CharField(
                        max_length=100, unique=True, verbose_name="Name"
                    ),
                ),
                (
                    "technique_type",
                    models.SmallIntegerField(
                        blank=True,
                        choices=[
                            (1, "Supervised"),
                            (2, "Unsupervised"),
                            (0, "Other"),
                        ],
                        default=0,
                        null=True,
                        verbose_name="Learning Technique Type",
                    ),
                ),
                (
                    "data_model",
                    models.CharField(
                        default="",
                        help_text='Django Model containing the data for the technique in the"app_label.model" format, i.e. "data_app.Data".',
                        max_length=100,
                        verbose_name="Data Model",
                    ),
                ),
                (
                    "learning_fields",
                    models.CharField(
                        blank=True,
                        help_text='Data Model\'s Fields to be used as input the technique separatedby comma and space, i.e. "avg1, rbc, myfield". If left blank, the fields defined in LEARNING_FIELDS in the Data Model will be used.',
                        max_length=255,
                        null=True,
                        verbose_name="Learning Fields",
                    ),
                ),
                (
                    "learning_fields_categorical",
                    models.CharField(
                        blank=True,
                        help_text='Subset of Learning Fields that are categorical data separatedby comma and space, i.e. "myfield". If left blank, the fields defined in LEARNING_FIELDS_CATEGORICAL in the Data Model will be used.',
                        max_length=255,
                        null=True,
                        verbose_name="Learning Fields Categorical",
                    ),
                ),
                (
                    "data_imputer",
                    models.CharField(
                        blank=True,
                        help_text='Data Imputer Class to handle possible Non-Available Values in dotted path format, i.e. "django_ai.ai_base.models.SimpleDataImputer".',
                        max_length=255,
                        null=True,
                        verbose_name="Data Imputer",
                    ),
                ),
                (
                    "data_imputer_object_id",
                    models.IntegerField(
                        blank=True,
                        help_text="Data Imputer Object id (internal use)",
                        null=True,
                        verbose_name="Data Imputer Object id",
                    ),
                ),
            ],
            options={
                "verbose_name": "Learning Technique",
                "verbose_name_plural": "Learning Techniques",
            },
        ),
        migrations.CreateModel(
            name="DataImputer",
            fields=[
                (
                    "learningtechnique_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="ai_base.learningtechnique",
                    ),
                ),
                (
                    "technique",
                    models.OneToOneField(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="data_imputer_object",
                        to="ai_base.learningtechnique",
                    ),
                ),
            ],
            options={
                "verbose_name": "Data Imputer",
                "verbose_name_plural": "Data Imputers",
            },
            bases=("ai_base.learningtechnique",),
        ),
        migrations.CreateModel(
            name="SimpleDataImputer",
            fields=[
                (
                    "dataimputer_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="ai_base.dataimputer",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
            bases=("ai_base.dataimputer",),
        ),
    ]
