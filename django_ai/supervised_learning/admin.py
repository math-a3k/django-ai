# -*- coding: utf-8 -*-

from django.contrib import admin

from .models import (HGBTree, SVC, )


@admin.register(SVC)
class SVCAdmin(admin.ModelAdmin):
    fieldsets = (
        ("General", {
            'fields': ('name', )
        }),
        ("Miscellanous", {
            'classes': ('collapse',),
            'fields': (
                ('engine_meta_iterations', 'engine_iterations'),
                ('counter', 'counter_threshold', 'threshold_actions'),
                ('engine_object_timestamp', 'image'),
                'metadata',
            ),
        }),
        ("Model Parameters", {
            'fields': (
                ('kernel', ),
                ('penalty_parameter', ),
                ('kernel_poly_degree', 'kernel_coefficient',
                    'kernel_independent_term', ),
                ('class_weight', ),
            )
        }),
        ("Implementation Parameters", {
            'fields': (
                ('decision_function_shape', ),
                ('estimate_probability', 'use_shrinking', ),
                ('tolerance', 'cache_size', 'random_seed', 'verbose', ),
            )
        })
    )

    def get_form(self, request, obj=None, **kwargs):  # pragma: no cover
        # Save obj reference in the request for future processing in Inline
        request._obj_ = obj
        form = super(SVCAdmin, self).get_form(request, obj, **kwargs)
        form.base_fields["metadata"].widget.attrs["disabled"] = "disabled"
        return(form)


@admin.register(HGBTree)
class HGBTreeAdmin(admin.ModelAdmin):
    fieldsets = (
        ("General", {
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
        ("Model Parameters", {
            'fields': (
                ('loss', ),
                ('learning_rate', ),
                ('max_leaf_nodes', 'max_depth',
                    'min_samples_leaf', 'l2_regularization',
                    'max_bins', ),
            )
        }),
        ("Implementation Parameters", {
            'fields': (
                ('warm_start', ),
                ('early_stopping', 'scoring', 'validation_fraction',
                    'n_iter_no_change', 'tol'),
                ('random_state', ),
                ('verbose', ),
            )
        })
    )

    def get_form(self, request, obj=None, **kwargs):  # pragma: no cover
        # Save obj reference in the request for future processing in Inline
        request._obj_ = obj
        form = super(HGBTreeAdmin, self).get_form(request, obj, **kwargs)
        form.base_fields["metadata"].widget.attrs["disabled"] = "disabled"
        return(form)
