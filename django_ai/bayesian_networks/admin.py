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
class BayesianNetworkAdmin(admin.ModelAdmin):
    readonly_fields = ['image', ]
    inlines = [
        BayesianNetworkNodeInline,
        BayesianNetworkEdgeInline,
    ]

    def get_form(self, request, obj=None, **kwargs):
        # Save obj reference in the request for future processing in Inline
        request._obj_ = obj
        return super(BayesianNetworkAdmin, self).get_form(request, obj, **kwargs)


# @admin.register(BayesianNetworkNode)
# class BayesianNetworkNodeAdmin(admin.ModelAdmin):
#     pass


# @admin.register(BayesianNetworkEdge)
# class BayesianNetworkNodeEdge(admin.ModelAdmin):
#     pass
