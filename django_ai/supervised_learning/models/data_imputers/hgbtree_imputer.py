from django_ai.supervised_learning.models.supervised_learning_imputer \
    import SupervisedLearningImputer


class HGBTreeImputer(SupervisedLearningImputer):

    def __init__(self, *args, **kwargs):
        kwargs['classification_technique'] = \
            'django_ai.supervised_learning.models.HGBTreeClassifier'
        kwargs['regression_technique'] = \
            'django_ai.supervised_learning.models.HGBTreeRegressor'
        super().__init__(*args, **kwargs)
