# -*- coding: utf-8 -*-
import os

from django.contrib import admin

from .models import (SpamFilter, )

if 'DJANGO_TEST' in os.environ:
    from django_ai.base.admin import DataColumnInline
else:  # pragma: no cover
    from base.admin import DataColumnInline


@admin.register(SpamFilter)
class SpamFilterAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {
            'fields': ('name', )
        }),
        ("Miscellanous", {
            'classes': ('collapse',),
            'fields': (
                ('engine_meta_iterations', 'engine_iterations'),
                ('counter', 'counter_threshold', 'threshold_actions'),
                ('engine_object_timestamp', ),
                'metadata',
            ),
        }),
        ("Spammable Model", {
            'fields': (
                'spam_model_is_enabled', 'spam_model_model',
            ),
        }),
        ("Labels", {
            'fields': (
                'labels_column',
            ),
        }),
        ("Classifier", {
            'fields': (
                'classifier',
            ),
        }),
        ("Cross Validation", {
            'fields': (
                'cv_is_enabled',
                ('cv_folds', 'cv_metric', ),
            ),
        }),
        ("Pre-Training", {
            'fields': (
                'pretraining',
            ),
        }),
        ("Bag of Words Transformation", {
            'fields': (
                ('bow_is_enabled', ),
                ('bow_use_tf_idf', ),
                ('bow_analyzer', 'bow_ngram_range_min',
                 'bow_ngram_range_max', ),
                ('bow_max_df', 'bow_min_df', 'bow_max_features', ),
            ),
        }),
        ("Bag of Words Transformation - Miscellanous", {
            'classes': ('collapse',),
            'fields': (
                ('bow_binary', ),
                ('bow_enconding', 'bow_decode_error', 'bow_strip_accents', ),
                ('bow_stop_words', 'bow_vocabulary', ),
            ),
        }),
    )

    inlines = [DataColumnInline, ]

    fieldsets_and_inlines_order = ('f', 'f', 'f', 'i', )

    def get_form(self, request, obj=None, **kwargs):  # pragma: no cover
        # Save obj reference in the request for future processing in Inline
        request._obj_ = obj
        form = super(SpamFilterAdmin, self).get_form(request, obj, **kwargs)
        form.base_fields["metadata"].widget.attrs["disabled"] = "disabled"
        return(form)
