# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-12-18 07:12
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('examples', '0015_sfptenron_sfptyoutube'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='sfptenron',
            options={
                'verbose_name': 'Spam Filter Pre-Training: Enron Email Data',
                'verbose_name_plural': ('Spam Filter Pre-Training: '
                                        'Enron Emails Data')},
        ),
        migrations.AlterModelOptions(
            name='sfptyoutube',
            options={
                'verbose_name':
                    'Spam Filter Pre-Training: Youtube Comment Data',
                'verbose_name_plural': ('Spam Filter Pre-Training: '
                                        'Youtube Comments Data')},
        ),
    ]
