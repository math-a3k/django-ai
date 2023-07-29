# Generated by Django 4.2 on 2023-04-13 04:44

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("supervised_learning", "0010_alter_hgbtreeregressor_loss"),
    ]

    operations = [
        migrations.AlterField(
            model_name="hgbtreeregressor",
            name="loss",
            field=models.CharField(
                blank=True,
                choices=[
                    ("squared_error", "Squared Error"),
                    ("least_absolute_deviation", "Least Absolute Deviation"),
                    ("poisson", "Poisson"),
                    ("absolute_error", "Absolute Error"),
                ],
                default="squared_error",
                help_text="The loss function to be used in the boosting process. If none is given, least_squares will be used.",
                max_length=50,
                null=True,
                verbose_name="Loss function",
            ),
        ),
    ]