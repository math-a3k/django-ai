from django.contrib import admin

from .models import EngineObjectModel, LearningTechnique


class EngineObjectAdmin(admin.ModelAdmin):
    model = EngineObjectModel

    fieldsets = (
        ("General", {
            'fields': ('name', )
        }),
        ("Engine Object", {
            'classes': ('collapse',),
            'fields': (
                ('is_inferred', 'engine_object_timestamp', 'metadata'),
            ),
        }),
        ("Automation", {
            'classes': ('collapse',),
            'fields': (
                ('counter', 'counter_threshold', 'threshold_actions'),
            ),
        }),
    )


class LearningTechniqueAdmin(admin.ModelAdmin):
    model = LearningTechnique

    data_model_fieldsets = ("Data Model", {
        'classes': ('collapse',),
        'fields': (
            ('data_model', ),
            ('learning_fields', 'learning_fields_categorical',),
        ),
    })

    data_imputer_fieldsets = ("Data Imputer", {
        'classes': ('collapse',),
        'fields': (
            ('data_imputer', ),
        ),
    })

    cift_fieldsets = ("CIFT", {
        'classes': ('collapse',),
        'fields': (
            ('cift_is_enabled', ),
        ),
    })

    fieldsets = EngineObjectAdmin.fieldsets + (
        data_model_fieldsets,
        data_imputer_fieldsets,
        cift_fieldsets
    )
