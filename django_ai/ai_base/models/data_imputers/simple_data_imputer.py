from numpy import nan as np_nan
from sklearn.impute import SimpleImputer as SKLSimpleImputer

from django.db.models.fields import BooleanField

from django_ai.ai_base.utils import allNotNone

from ..data_imputer import DataImputer


class SimpleDataImputer(DataImputer):
    NA_COLUMN_FILL = 0
    engine_object_class = SKLSimpleImputer
    _non_na_fields = None

    def engine_object_init(self):
        field_imputer = {}
        learning_fields_categorical = (
            self._get_data_learning_fields_categorical()
        )
        for field in self.get_non_na_fields():
            if field in learning_fields_categorical:
                field_imputer[field] = self.engine_object_class(
                    strategy="most_frequent"
                )
            else:
                field_imputer[field] = self.engine_object_class(
                    strategy="median"
                )
        return field_imputer

    def get_inference_scores(self):
        return None

    def get_engine_object_conf(self):
        return None

    def engine_object_perform_inference(self):
        data = self.get_data()
        eo = self.get_engine_object()
        for field in self.get_non_na_fields():
            eo[field].fit(data[field])
        return eo

    def engine_object_predict(self, field):
        eo = self.get_engine_object()
        value = eo[field].transform([[np_nan]])[0][0]
        return value

    def get_data(self):
        data_list = [
            list(row)
            for row in self.technique._get_data_queryset().values_list(
                *self.get_supported_fields()
            )
        ]
        transposed_data_dict = {
            field: [] for field in self.get_supported_fields()
        }
        for row in data_list:
            row_dict = self.technique._observation_list_to_dict(row)
            for field in row_dict:
                if row_dict[field] is not None:
                    if isinstance(row_dict[field], bool):
                        row_dict[field] = int(row_dict[field])
                    transposed_data_dict[field].append([row_dict[field]])
        return transposed_data_dict

    def impute_row(self, row):
        if not allNotNone(row):
            row = self.technique._observation_list_to_dict(row)
            for field in row:
                if row[field] is None:
                    if field in self.get_non_na_fields():
                        row[field] = self.engine_object_predict(field)
                        if row[field] in [0, 1]:
                            data_model = self._get_data_model()
                            django_field = data_model._meta.get_field(field)
                            if isinstance(django_field, BooleanField):
                                row[field] = bool(row[field])
                    else:
                        row[field] = self.NA_COLUMN_FILL
            return self.technique._observation_dict_to_list(row)
        else:
            return row

    def get_non_na_fields(self):
        if not self._non_na_fields:
            self._non_na_fields = [
                f
                for f in self.get_supported_fields()
                if f not in self.technique._get_data_learning_fields_na()
            ]
        return self._non_na_fields

    def get_supported_fields(self):
        return self.technique._get_data_learning_fields_supported()
