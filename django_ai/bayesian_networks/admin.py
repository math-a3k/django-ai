# coding: utf-8

from django.contrib import admin
from nested_admin import (NestedModelAdmin, NestedStackedInline,
                          NestedTabularInline)

from .models import (BayesianNetwork, BayesianNetworkNode,
                     BayesianNetworkNodeColumn, BayesianNetworkEdge)


class BayesianNetworkNodeColumnInline(NestedTabularInline):
    model = BayesianNetworkNodeColumn
    sortable_field_name = "position"
    fields = ["ref_model", "ref_column", "position"]
    extra = 1


class BayesianNetworkNodeInline(NestedStackedInline):
    model = BayesianNetworkNode
    extra = 1
    inlines = [BayesianNetworkNodeColumnInline, ]
    fieldsets = (
        (None, {
            'fields': ('name', 'node_type',)
        }),
        ("Stochastic Type", {
            'fields': (('distribution', 'distribution_params'),
                       'is_observable', ),
        }),
        ("Deterministic Type", {
            'fields': (('deterministic', 'deterministic_params'), ),
        }),
        ("Visualization", {
            'classes': ('collapse',),
            'fields': (('graph_interval', 'image'), ),
        }),
        ("Timestamps", {
            'classes': ('collapse',),
            'fields': (('engine_object_timestamp',
                        'engine_inferred_object_timestamp'), ),
        }),
    )

    class Media:
        css = {
            'all': ('/static/css/admin/bayesian_networks.css',)
        }


class BayesianNetworkEdgeInline(NestedTabularInline):
    model = BayesianNetworkEdge
    extra = 1

    def formfield_for_foreignkey(self, db_field, request=None, **kwargs):
        field = super(BayesianNetworkEdgeInline, self)\
            .formfield_for_foreignkey(db_field, request, **kwargs)
        # Display only Nodes from the Network or None
        if db_field.name in ['child', 'parent']:
            if request._obj_ is not None:
                field.queryset = field.queryset.filter(network=request._obj_)
            else:
                field.queryset = field.queryset.none()
        return field


@admin.register(BayesianNetwork)
class BayesianNetworkAdmin(NestedModelAdmin):
    fieldsets = (
        (None, {
            'fields': ('name', 'network_type', 'results_storage')
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
    )
    inlines = [
        BayesianNetworkNodeInline,
        BayesianNetworkEdgeInline,
    ]

    def get_form(self, request, obj=None, **kwargs):
        # Save obj reference in the request for future processing in Inline
        request._obj_ = obj
        form = super(BayesianNetworkAdmin, self).get_form(request, obj,
                                                          **kwargs)
        form.base_fields["metadata"].widget.attrs["disabled"] = "disabled"
        return(form)

# @admin.register(BayesianNetworkNode)
# class BayesianNetworkNodeAdmin(admin.ModelAdmin):
#     pass


# @admin.register(BayesianNetworkEdge)
# class BayesianNetworkNodeEdge(admin.ModelAdmin):
#     pass
