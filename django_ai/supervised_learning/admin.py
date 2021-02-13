from django_ai.ai_base.admin import EngineObjectAdmin, LearningTechniqueAdmin


class SupervisedLearningTechniqueAdmin(LearningTechniqueAdmin):

    data_model_fieldsets = ("Data Model", {
        'classes': ('collapse',),
        'fields': (
            ('data_model', ),
            ('learning_fields', 'learning_fields_categorical',),
            ('learning_target', ),
            ('monotonic_constraints', ),
        ),
    })

    cross_validation_fieldsets = ("Cross Validation", {
        'classes': ('collapse',),
        'fields': (
            'cv_is_enabled',
            ('cv_folds', 'cv_metrics', ),
        ),
    })

    fieldsets = EngineObjectAdmin.fieldsets + (
        data_model_fieldsets,
        LearningTechniqueAdmin.data_imputer_fieldsets,
        LearningTechniqueAdmin.cift_fieldsets,
        cross_validation_fieldsets,
    )


class HGBTreeClassifierAdmin(SupervisedLearningTechniqueAdmin):

    engine_parameters_fieldsets = ("Engine Parameters", {
        'classes': ('collapse',),
        'fields': (
            ('loss', ),
            ('learning_rate', ),
            ('max_leaf_nodes', 'max_depth',
                'min_samples_leaf', 'l2_regularization',
                'max_bins', ),
            ('max_iter', ),
            ('warm_start', ),
            ('early_stopping', 'scoring', 'validation_fraction',
                'n_iter_no_change', 'tol', ),
            ('random_state', ),
            ('verbose', ),

        ),
    })

    fieldsets = SupervisedLearningTechniqueAdmin.fieldsets + (
        engine_parameters_fieldsets,
    )


class SVCAdmin(SupervisedLearningTechniqueAdmin):

    engine_parameters_fieldsets = ("Engine Parameters", {
        'classes': ('collapse',),
        'fields': (
            ('kernel', ),
            ('penalty_parameter', ),
            ('kernel_poly_degree', 'kernel_coefficient',
                'kernel_independent_term', ),
            ('class_weight', ),
            ('decision_function_shape', ),
            ('estimate_probability', 'use_shrinking', ),
            ('tolerance', 'cache_size', 'random_state', 'verbose', ),
        ),
    })

    fieldsets = SupervisedLearningTechniqueAdmin.fieldsets + (
        engine_parameters_fieldsets,
    )


class SVRAdmin(SupervisedLearningTechniqueAdmin):

    engine_parameters_fieldsets = ("Engine Parameters", {
        'classes': ('collapse',),
        'fields': (
            ('kernel', ),
            ('penalty_parameter', ),
            ('kernel_poly_degree', 'kernel_coefficient',
                'kernel_independent_term', ),
            ('use_shrinking', ),
            ('tolerance', 'cache_size', 'verbose', ),
        ),
    })

    fieldsets = SupervisedLearningTechniqueAdmin.fieldsets + (
        engine_parameters_fieldsets,
    )
