# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-05-02 23:12
from __future__ import unicode_literals
import numpy as np
import random

from django.db import migrations


def populate_userinfos(apps, schema_editor):
    UserInfo = apps.get_model("sl_test_models", "UserInfo")
    # Use a fixed seed for generate content
    np.random.seed(123456)
    random.seed(123456)
    # Size of table
    size = 200
    # Sex is ~ 70% F (0) / 30% M (1)
    sex = np.random.binomial(1, 0.7, size)  # 200 Bernoullies :)
    sex = ["xx" if s else "xy" for s in sex]
    # Age is around 30, mostly between 25 and 35
    age = np.floor(np.random.normal(30, 2, size=(size,)))
    # Average 1 is a metric normally distributed around 10 with a std dev of 5
    avg1 = np.random.normal(10, 5, size=(size,))
    avg1[-3] = None
    avg1[-1] = None
    # Cluster 2
    cluster_2 = [
        random.sample([0, 1, 2], counts=[1, 5, 1], k=1)[0] for r in range(size)
    ]
    cluster_2[-2] = None
    cluster_2[-1] = None
    # Boolean Field is ~ 50% 0 / 50% 1
    bool_field = list(np.random.binomial(1, 0.5, size))
    bool_field[-2] = None
    bool_field[-1] = None

    # Create the objects in the Model
    uis = []
    for i in range(0, size):
        uis.append(
            UserInfo(
                age=age[i],
                sex=sex[i],
                avg1=avg1[i],
                cluster_2=cluster_2[i],
                bool_field=bool_field[i],
            )
        )
    UserInfo.objects.bulk_create(uis)
    UserInfo2 = apps.get_model("sl_test_models", "UserInfo2")
    # Use a fixed seed for generate content
    np.random.seed(123456)
    # Size of table
    size = 100
    # Average 2 is a metric normally distributed around 20 with a std dev of 5
    avg2 = np.random.normal(20, 5, size=(size,))
    # Average Times spent on Pages B is a metric normally distributed around
    # 30 with a std dev of 5
    avg_time_pages_b = np.random.normal(30, 5, size=(size,))
    # Create the objects in the Model
    uis = []
    for i in range(0, size):
        uis.append(
            UserInfo2(avg_time_pages_b=avg_time_pages_b[i], avg2=avg2[i])
        )
    UserInfo2.objects.bulk_create(uis)


def unpopuplate_userinfos(apps, schema_editor):
    UserInfo = apps.get_model("sl_test_models", "UserInfo")
    UserInfo.objects.all().delete()
    UserInfo2 = apps.get_model("sl_test_models", "UserInfo2")
    UserInfo2.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ("sl_test_models", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(populate_userinfos, unpopuplate_userinfos),
    ]
