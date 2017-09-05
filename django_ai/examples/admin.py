# -*- coding: utf-8 -*-

from django.contrib import admin

from .models import UserInfo


@admin.register(UserInfo)
class UserInfoAdmin(admin.ModelAdmin):
    list_display = ['id', 'age', 'get_sex_display', 'avg1']
