# coding: utf-8

from django.contrib import admin

from .models import (BayesianNetwork, BayesianNetworkNode,
                     BayesianNetworkEdge)


class BayesianNetworkNodeInline(admin.StackedInline):
    model = BayesianNetworkNode
    extra = 1
    fieldsets = (
        (None, {
            'fields': ('name', 'node_type',)
        }),
        ("Stochastic Type", {
            'fields': (('distribution', 'distribution_params'),
                        'is_observable', ('ref_model', 'ref_column'), ),
        }),
        ("Deterministic Type", {
            'fields': (('deterministic', 'deterministic_params'), ),
        }),
        ("Visualization", {
            'fields': (('graph_interval', 'image'), ),
        }),
        ("Timestamps", {
            'classes': ('collapse',),
            'fields': (('engine_object_timestamp',
                        'engine_inferred_object_timestamp'), ),
        }),
    )


class BayesianNetworkEdgeInline(admin.TabularInline):
    model = BayesianNetworkEdge
    extra = 1


@admin.register(BayesianNetwork)
class BayesianNetworkAdmin(admin.ModelAdmin):
    readonly_fields = ['image', ]
    inlines = [
        BayesianNetworkNodeInline,
        BayesianNetworkEdgeInline,
    ]


# @admin.register(BayesianNetworkNode)
# class BayesianNetworkNodeAdmin(admin.ModelAdmin):
#     pass


# @admin.register(BayesianNetworkEdge)
# class BayesianNetworkNodeEdge(admin.ModelAdmin):
#     pass
