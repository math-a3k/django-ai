from django_ai.supervised_learning.models.supervised_learning_imputer import (
    SupervisedLearningImputer,
)


class SVMImputer(SupervisedLearningImputer):
    def __init__(self, *args, **kwargs):
        kwargs[
            "classification_technique"
        ] = "django_ai.supervised_learning.models.svm.SVC"
        kwargs[
            "regression_technique"
        ] = "django_ai.supervised_learning.models.svm.SVR"
        super().__init__(*args, **kwargs)
