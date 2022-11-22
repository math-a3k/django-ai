# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-05-02 23:12
from __future__ import unicode_literals
import numpy as np

from django.db import migrations


def populate_userinfos(apps, schema_editor):
    UserInfo = apps.get_model("test_models", "UserInfo")
    # Use a fixed seed for generate content
    np.random.seed(123456)
    # Size of table
    size = 200
    # Sex is ~ 70% F (0) / 30% M (1)
    sex = np.random.binomial(1, 0.7, size)  # 200 Bernoullies :)
    # Age is around 30, mostly between 25 and 35
    age = np.floor(np.random.normal(30, 2, size=(size,)))
    # Average 1 is a metric normally distributed around 10 with a std dev of 5
    avg1 = np.random.normal(10, 5, size=(size,))
    # Create the objects in the Model
    uis = []
    for i in range(0, size):
        uis.append(UserInfo(age=age[i], sex=sex[i], avg1=avg1[i]))
    UserInfo.objects.bulk_create(uis)


def unpopuplate_userinfos(apps, schema_editor):
    UserInfo = apps.get_model("test_models", "UserInfo")
    UserInfo.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('test_models', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(populate_userinfos,
                             unpopuplate_userinfos),
    ]