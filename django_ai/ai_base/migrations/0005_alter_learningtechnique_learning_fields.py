# Generated by Django 4.2 on 2023-04-28 00:11

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ai_base", "0004_alter_learningtechnique_id"),
    ]

    operations = [
        migrations.AlterField(
            model_name="learningtechnique",
            name="learning_fields",
            field=models.CharField(
                blank=True,
                help_text='Data Model\'s Fields to be used as input the technique separatedby comma and space, i.e. "avg1, rbc, myfield". If left blank, the fields defined in LEARNING_FIELDS in the Data Model will be used.',
                max_length=2550,
                null=True,
                verbose_name="Learning Fields",
            ),
        ),
    ]