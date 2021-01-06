# -*- coding: utf-8 -*-

from django.contrib.contenttypes.admin import GenericTabularInline

from .models import DataColumn


class DataColumnInline(GenericTabularInline):  # pragma: no cover
    model = DataColumn
    # sortable_field_name = "position"
    fields = ["ref_model", "ref_column", "position"]
    ct_field = 'content_type'
    ct_fk_field = 'object_id'
    extra = 1
