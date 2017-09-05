# coding: utf-8

from django.contrib import admin

from .models import (BayesianNetwork, BayesianNetworkNode, BayesianNetworkEdge)


class BayesianNetworkNodeInline(admin.StackedInline):
    model = BayesianNetworkNode
    extra = 1


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
