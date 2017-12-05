# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-11-25 07:27
from __future__ import unicode_literals
import os
import tarfile
import urllib.request
import random

from django.db import migrations, models

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_FILE_NAME = os.path.join(CURRENT_DIR, "enron1.tar.gz")
SAMPLE_DATA_FILE_URL = ("http://www.aueb.gr/users/ion/data/enron-spam/"
                        "preprocessed/enron1.tar.gz")


def confirm(question):
    """
    https://gist.github.com/garrettdreyfus/8153571
    """
    reply = str(input(' -> ' + question + ' (Y/n): ')).lower().strip()
    if reply == 'y' or reply == '':
        return True
    elif reply == 'n':
        return False
    else:
        return confirm("Mmmm... Please enter")


def download_and_process_sample_data(apps, schema_editor):
    """
    Forward Operation: Downloads if neccesary the sample data and populates
    the CommentOfMySite model.
    """
    CommentOfMySite = apps.get_model("examples",
                                     "CommentOfMySite")
    random.seed(1234567)
    # -> Download if not exists
    if not os.path.exists(SAMPLE_DATA_FILE_NAME):
        if confirm("Proceed to download Enron sample data?"):
            urllib.request.urlretrieve(
                SAMPLE_DATA_FILE_URL,
                SAMPLE_DATA_FILE_NAME
            )
    # -> Process the file
    with tarfile.open(name=SAMPLE_DATA_FILE_NAME, mode="r:gz") as tfile:
        total_files = len(tfile.getnames()) - 4
        for member in tfile.getmembers():
            if member.isfile():
                message = tfile.extractfile(member).read()
                CommentOfMySite.objects.create(
                    user_id=random.randint(1, total_files),
                    comment=message.decode('raw_unicode_escape'),
                    is_spam=("spam" in member.name),
                    is_misclassified=False,
                    is_revised=True
                )


def confirm_deletion_sample_data_file(apps, schema_editor):
    """
    Backward Operation: Remove the file if deemed necessary, no need to remove
    the objects as the table will be removed.
    """
    if not confirm("Leave downloaded Enron sample data file for the future?"):
        os.remove(SAMPLE_DATA_FILE_NAME)


class Migration(migrations.Migration):

    dependencies = [
        ('examples', '0012_update_clustering_bn'),
    ]

    operations = [
        migrations.CreateModel(
            name='CommentOfMySite',
            fields=[
                ('id', models.AutoField(
                    auto_created=True,
                    primary_key=True, serialize=False,
                    verbose_name='ID')
                 ),
                ('is_spam', models.BooleanField(
                    default=False,
                    help_text='If the object is Spam',
                    verbose_name='Is Spam?')
                 ),
                ('is_misclassified', models.BooleanField(
                    default=False,
                    help_text=('If the object has been misclassified by '
                               'the Spam Filter'),
                    verbose_name='Is Misclassified?')
                 ),
                ('is_revised', models.BooleanField(
                    default=False,
                    help_text=('If the object classification has been revised '
                               'by a Human'),
                    verbose_name='Is Revised?')
                 ),
                ('comment', models.TextField(
                    verbose_name='Comment')
                 ),
                ('user_id', models.SmallIntegerField(
                    verbose_name='User ID')
                 ),
            ],
            options={
                'verbose_name': 'Comment of my Site',
                'verbose_name_plural': 'Comments of my Site',
            },
        ),
        migrations.RunPython(
            download_and_process_sample_data,
            confirm_deletion_sample_data_file
        ),
    ]