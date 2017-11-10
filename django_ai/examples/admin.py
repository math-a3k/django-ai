# -*- coding: utf-8 -*-

from django.contrib import admin

from .models import UserInfo

UserInfo.get_sex_display.short_description = "Sex"
@admin.register(UserInfo)
class UserInfoAdmin(admin.ModelAdmin):
    list_display = ['id', 'age', 'get_sex_display', 'avg1',
                    'avg_time_pages', 'visits_pages', 'avg_time_pages_a',
                    'visits_pages_a', 'cluster_1']
    fieldsets = (
        (None, {
            'fields': ('age', 'sex',)
        }),
        (None, {
            'fields': ('avg1',)
        }),
        ("Pages of Type X", {
            'fields': (
                        ('visits_pages_a', 'avg_time_pages_a', ),
                        ('visits_pages_b', 'avg_time_pages_b', ),
                        ('visits_pages_c', 'avg_time_pages_c', ),
                        ('visits_pages_d', 'avg_time_pages_d', ),
                        ('visits_pages_e', 'avg_time_pages_e', ),
                        ('visits_pages_f', 'avg_time_pages_f', ),
                        ('visits_pages_g', 'avg_time_pages_g', ),
                        ('visits_pages_h', 'avg_time_pages_h', ),
                        ('visits_pages_i', 'avg_time_pages_i', ),
                        ('visits_pages_j', 'avg_time_pages_j', ),
                      ),
        }),
        ("Visits and Pages (General)", {
            'fields': (
                        ('visits_pages', 'avg_time_pages', ),
                      ),
        }),
    )
