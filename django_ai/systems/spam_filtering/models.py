# -*- coding: utf-8 -*-

import os
from itertools import (chain, )
from pickle import HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL

from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import (ValidationError, ImproperlyConfigured)
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone

import numpy as np
from picklefield.fields import PickledObjectField
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.model_selection import (cross_val_score, )
from scipy.sparse import csr_matrix

if 'DJANGO_TEST' in os.environ:
    from django_ai.base.models import SupervisedLearningTechnique
    from django_ai.base.utils import (get_model, )
else:  # pragma: no cover
    from base.models import (SupervisedLearningTechnique, )
    from base.utils import (get_model, )


class SpamFilter(SupervisedLearningTechnique):
    """
    Main object for the Spam Filtering System.
    """
    #: BoW Decode Error choices
    BOW_DECODE_ERROR_CHOICES = (
        ('strict', _('Strict')),
        ('ignore', _('Ignore')),
        ('replace', _('Replace')),
    )
    #: BoW Strip Accent choices
    BOW_STRIP_ACCENTS_CHOICES = (
        ('ascii', 'ASCII'),
        ('unicode', 'Unicode'),
    )
    #: BoW Analyzer units choices
    BOW_ANALYZER_CHOICES = (
        ('word', _('Word')),
        ('char', _('Character')),
        ('char_wb', _("Characters in Word-Boundaries")),
    )
    #: Cross Validation Available Metrics choices
    CV_CHOICES = (
        ('accuracy', _("Accuracy")),
        ('average_precision', _("Average Precision")),
        ('f1', _("F1")),
        ('neg_log_loss', _("Logistic Loss")),
        ('precision', _("Precision")),
        ('recall', _("Recall")),
        ('roc_auc', _("Area under ROC Curve")),
    )
    #: Engine Object Vectorizer
    engine_object_vectorizer = PickledObjectField(
        "Engine Object Vectorizer",
        protocol=pickle_HIGHEST_PROTOCOL,
        blank=True, null=True
    )
    #: Engine Object Data
    engine_object_data = PickledObjectField(
        "Engine Object Data",
        protocol=pickle_HIGHEST_PROTOCOL,
        blank=True, null=True
    )
    #: Classifier to be used in the System, in the
    #: "app_label.model|name" format, i.e.
    #: "supervised_learning.SVC|My SVM"
    classifier = models.CharField(
        "Supervised Learning Classifier",
        max_length=100, blank=True, null=True,
        help_text=(
            'Classifier to be used in the System, in the '
            '"app_label.model|name" format, i.e. '
            '"supervised_learning.SVC|My SVM"'
        )
    )
    #: Whether to use a Spammable Model as a data source
    spam_model_is_enabled = models.BooleanField(
        "Use a Spammable Model?",
        default=True,
        help_text=(
            'Use a Spammable Model'
        )
    )
    #: "IsSpammable-Django Model" to be used with the Spam Filter (in
    #: the "app_label.model" format, i.e. "examples.CommentOfMySite")
    spam_model_model = models.CharField(
        "Spammable Django Model",
        max_length=100, blank=True, null=True,
        help_text=(
            '"IsSpammable-Django Model" to be used with the Spam Filter (in '
            'the "app_label.model" format, i.e. "examples.CommentOfMySite")'
        )
    )
    # -> Cross Validation
    #: Metric to be evaluated in Cross Validation
    cv_metric = models.CharField(
        "Cross Validation Metric",
        max_length=20, blank=True, null=True, choices=CV_CHOICES,
        help_text=(
            'Metric to be evaluated in Cross Validation'
        )
    )
    # -> Bag of Words Transformation
    #: Enable Bag of Words transformation
    bow_is_enabled = models.BooleanField(
        "Enable Bag of Words representation?",
        default=True,
        help_text=(
            'Enable Bag of Words transformation'
        )
    )
    # (skl) encoding : string, ‘utf-8’ by default.
    #: Encoding to be used to decode the corpus
    bow_enconding = models.CharField(
        "(BoW) Encoding",
        default='utf-8', max_length=20,
        help_text=(
            'Encoding to be used to decode.'
        )
    )
    # (skl) decode_error : {‘strict’, ‘ignore’, ‘replace’}
    #: Instruction on what to do if a byte sequence is given to
    #: analyze that contains characters not of the given encoding.
    #: By default, it is ‘strict’, meaning that a UnicodeDecodeError
    #: will be raised. Other values are ‘ignore’ and ‘replace’.'
    bow_decode_error = models.CharField(
        "(BoW) Decode Error",
        default='strict', max_length=20, choices=BOW_DECODE_ERROR_CHOICES,
        help_text=_((
            'Instruction on what to do if a byte sequence is given to '
            'analyze that contains characters not of the given encoding. '
            'By default, it is ‘strict’, meaning that a UnicodeDecodeError '
            'will be raised. Other values are ‘ignore’ and ‘replace’.'
        ))
    )
    # (skl) strip_accents : {‘ascii’, ‘unicode’, None}
    #: Remove accents during the preprocessing step. ‘ascii’ is a fast
    #: method that only works on characters that have an direct ASCII
    #: mapping. ‘unicode’ is a slightly slower method that works on
    #: any characters. None (default) does nothing.
    bow_strip_accents = models.CharField(
        "(BoW) Strip Accents",
        default=None, max_length=20, choices=BOW_STRIP_ACCENTS_CHOICES,
        blank=True, null=True,
        help_text=_((
            'Remove accents during the preprocessing step. ‘ascii’ is a fast '
            'method that only works on characters that have an direct ASCII '
            'mapping. ‘unicode’ is a slightly slower method that works on '
            'any characters. None (default) does nothing.'
        ))
    )
    # (skl) analyzer : string, {‘word’, ‘char’, ‘char_wb’} or callable
    #: Whether the feature should be made of word or character n-grams.
    #: Option ‘Chars in W-B’ creates character n-grams only from text inside
    #: word boundaries; n-grams at the edges of words are padded with
    #: space.'
    bow_analyzer = models.CharField(
        "(BoW) Analyzer",
        default='word', max_length=20, choices=BOW_ANALYZER_CHOICES,
        help_text=_((
            'Whether the feature should be made of word or character n-grams. '
            'Option ‘Chars in W-B’ creates character n-grams only from text '
            'inside word boundaries; n-grams at the edges of words are padded '
            'with space.'
        ))
    )
    # (skl) ngram_range : tuple (min_n, max_n)
    #: The lower boundary of the range of n-values for
    #: different n-grams to be extracted. All value
    #: of n such that min_n <= n <= max_n will be used.
    bow_ngram_range_min = models.SmallIntegerField(
        "(BoW) n-gram Range - Min",
        default=1,
        help_text=_((
            'The lower boundary of the range of n-values for '
            'different n-grams to be extracted. All values '
            'of n such that min_n <= n <= max_n will be used.'
        ))
    )
    #: The upper boundary of the range of n-values for
    #: different n-grams to be extracted. All values
    #: of n such that min_n <= n <= max_n will be used.
    bow_ngram_range_max = models.SmallIntegerField(
        "(BoW) n-gram Range - Max",
        default=1,
        help_text=_((
            'The upper boundary of the range of n-values for '
            'different n-grams to be extracted. All values '
            'of n such that min_n <= n <= max_n will be used.'
        ))
    )
    # (skl) stop_words : string {‘english’}, list, or None (default)
    #: If ‘english’, a built-in stop word list for English is used.
    #: If a comma-separated string, that list is assumed to contain
    #: stop words, all of which will be removed from the resulting
    #: tokens. Only applies if analyzer == ´word´. If None, no stop
    #: words will be used. max_df can be set to a value in the range
    #: [0.7, 1.0) to automatically detect and filter stop words based
    #: on intra corpus document frequency of terms.'
    bow_stop_words = models.TextField(
        "(BoW) Stop Words",
        default=None, blank=True, null=True,
        help_text=_((
            'If ‘english’, a built-in stop word list for English is used. '
            'If a comma-separated string, that list is assumed to contain '
            'stop words, all of which will be removed from the resulting '
            'tokens. Only applies if analyzer == ´word´. If None, no stop '
            'words will be used. max_df can be set to a value in the range '
            '[0.7, 1.0) to automatically detect and filter stop words based '
            'on intra corpus document frequency of terms.'
        ))
    )
    # (skl) max_df : float in range [0.0, 1.0] or int, default=1.0
    # When building the vocabulary ignore terms that have a document
    # frequency strictly higher than the given threshold
    # (corpus-specific stop words). If float, the parameter represents
    # a proportion of documents, integer absolute counts. This
    # parameter is ignored if vocabulary is not None.
    bow_max_df = models.FloatField(
        "(BoW) Maximum Document Frequency",
        default=1.0,
        help_text=_((
            'When building the vocabulary ignore terms that have a document '
            'frequency strictly higher than the given threshold '
            '(corpus-specific stop words). If float, the parameter represents '
            'a proportion of documents, integer absolute counts. This '
            'parameter is ignored if vocabulary is not None.'
        ))
    )
    # (skl) min_df : float in range [0.0, 1.0] or int, default=1
    #: When building the vocabulary ignore terms that have a document
    #: frequency strictly lower than the given threshold. This value is
    #: also called cut-off in the literature. If float, the parameter
    #: represents a proportion of documents, integer absolute counts.
    #: This parameter is ignored if vocabulary is not None.
    bow_min_df = models.FloatField(
        "(BoW) Minimum Document Frequency",
        default=1,
        help_text=_((
            'When building the vocabulary ignore terms that have a document '
            'frequency strictly lower than the given threshold. This value is '
            'also called cut-off in the literature. If float, the parameter '
            'represents a proportion of documents, integer absolute counts. '
            'This parameter is ignored if vocabulary is not None.'
        ))
    )
    # (skl) max_features : int or None, default=None
    #: If not None, build a vocabulary that only consider the top
    #: max_features ordered by term frequency across the corpus.
    #: This parameter is ignored if vocabulary is not None.
    bow_max_features = models.IntegerField(
        "(BoW) Maximum Features",
        default=None, blank=True, null=True,
        help_text=_((
            'If not None, build a vocabulary that only consider the top '
            'max_features ordered by term frequency across the corpus. '
            ' This parameter is ignored if vocabulary is not None.'
        ))
    )
    # (skl) vocabulary : Mapping or iterable, optional
    #: A Mapping (e.g., a dict) where keys are terms and values
    #: are indices in the feature matrix.
    #: If not given, a vocabulary is determined from the input
    #: documents. Indices in the mapping should not be repeated and
    #: should not have any gap between 0 and the largest index.
    bow_vocabulary = models.TextField(
        "(BoW) Vocabulary",
        default=None, blank=True, null=True,
        help_text=_((
            'A Mapping (e.g., a dict) where keys are terms and values '
            'are indices in the feature matrix. '
            'If not given, a vocabulary is determined from the input '
            'documents. Indices in the mapping should not be repeated and '
            'should not have any gap between 0 and the largest index.'
        ))
    )
    # (skl) binary : boolean, default=False
    #: If True, all non zero counts are set to 1. This is useful for
    #: discrete probabilistic models that model binary events rather
    #: than integer counts.
    bow_binary = models.BooleanField(
        "(BoW) Use Binary representation?",
        default=False,
        help_text=_((
            'If True, all non zero counts are set to 1. This is useful for '
            'discrete probabilistic models that model binary events rather '
            'than integer counts.'
        ))
    )
    #: Use the TF-IDF transformation?
    bow_use_tf_idf = models.BooleanField(
        "(BoW) Use the TF-IDF transformation?",
        default=True,
        help_text=_((
            'Use the TF-IDF transformation?'
        ))
    )

    class Meta:
        verbose_name = "Spam Filter"
        verbose_name_plural = "Spam Filters"
        # app_label = "systems.spam_filtering"

    def save(self, *args, **kwargs):
        # Initialize metadata field if corresponds
        if self.metadata == {}:
            self.metadata["current_inference"] = {}
            self.metadata["previous_inference"] = {}

        super(SpamFilter, self).save(*args, **kwargs)

    def __str__(self):
        return("[Spam Filter] {}".format(self.name))

    def clean(self):
        if self.classifier:
            # Check the validity of the Classifier
            try:
                app_model, object_name = self.classifier.split("|")
                app, model = app_model.split(".")
            except Exception:
                raise ValidationError({'classifier': _(
                    'Invalid format'
                )})
            try:
                model_class = ContentType.objects.get(
                    app_label=app,
                    model=model.lower()
                ).model_class()
            except Exception:
                raise ValidationError({'classifier': _(
                    'The App and Model must be a valid Django App and Model'
                )})
            try:
                model_class.objects.get(name=object_name)
            except Exception:
                raise ValidationError({'classifier': _(
                    'Cannot get the object "{}" from the '
                    '{} model'.format(
                        object_name, model_class._meta.verbose_name)
                )})
        if self.pretraining:
            # Check the validity of the Pretraining field
            try:
                app, model = self.pretraining.split(".")
            except Exception:
                raise ValidationError({'pretraining': _(
                    'Invalid format'
                )})
            try:
                model_class = ContentType.objects.get(
                    app_label=app,
                    model=model.lower()
                ).model_class()
            except Exception:
                raise ValidationError({'classifier': _(
                    'The App and Model must be a valid Django App and Model'
                )})
        if self.spam_model_is_enabled:
            # Check the validity of the Spammable Model field
            try:
                app, model = self.spam_model_model.split(".")
            except Exception:
                raise ValidationError({'spam_model_model': _(
                    'Invalid format'
                )})
            try:
                model_class = ContentType.objects.get(
                    app_label=app,
                    model=model.lower()
                ).model_class()
            except Exception:
                raise ValidationError({'spam_model_model': _(
                    'The App and Model must be a valid Django App and Model'
                )})

        super(SpamFilter, self).clean()

    def get_pretraining_data(self):
        if self.pretraining:
            model = get_model(self.pretraining)
            pt_data = model.objects.values_list(model.SPAMMABLE_FIELD,
                                                flat=True)
            return(list(pt_data))
        else:
            return(None)

    def get_pretraining_labels(self):
        if self.pretraining:
            model = get_model(self.pretraining)
            pt_labels = model.objects.values_list(model.SPAM_LABEL_FIELD,
                                                  flat=True)
            return(list(pt_labels))
        else:
            return(None)

    def get_data(self, utf8_point_repr=False):
        if self.spam_model_is_enabled:
            model = get_model(self.spam_model_model)
            data = list(model.objects.values_list(model.SPAMMABLE_FIELD,
                                                  flat=True))
        else:
            data = super(SpamFilter, self).get_data()
            # Flatten list
            data = list(chain.from_iterable(data))
        if self.pretraining:
            data += self.get_pretraining_data()
        if utf8_point_repr:
            max_length = max([len(text) for text in data])
            data = [[ord(character) for character in text.ljust(max_length)]
                    for text in data]
        return(data)

    def get_labels(self):
        if self.spam_model_is_enabled:
            model = get_model(self.spam_model_model)
            labels = list(model.objects.values_list(model.SPAM_LABEL_FIELD,
                                                    flat=True))
        else:
            labels = super(SpamFilter, self).get_labels()
        if self.pretraining:
            labels = list(labels)
            labels += self.get_pretraining_labels()
        return(labels)

    def get_classifier(self):
        app_model, object_name = self.classifier.split("|")
        app, model = app_model.split(".")
        model_class = ContentType.objects.get(
            app_label=app,
            model=model.lower()
        ).model_class()
        return(model_class.objects.get(name=object_name))

    def get_engine_object_vectorizer(self, reconstruct=False, save=True):
        """
        Retrieves / Initializes the Engine's Vectorizer and transforms the
        data making it available in the `self.engine_object_data` field.
        """
        if self.engine_object_vectorizer is not None and not reconstruct:
            return(self.engine_object_vectorizer)
        else:
            if self.bow_use_tf_idf:
                BoW_Vectorizer = TfidfVectorizer
            else:
                BoW_Vectorizer = CountVectorizer
            bow_vectorizer_args = {
                'encoding': self.bow_enconding,
                'decode_error': self.bow_decode_error,
                'strip_accents': self.bow_strip_accents,
                'ngram_range': (self.bow_ngram_range_min,
                                self.bow_ngram_range_max),
                'stop_words': self.bow_stop_words,
                'max_df': self.bow_max_df,
                'min_df': self.bow_min_df,
                'max_features': self.bow_max_features,
                'vocabulary': self.bow_vocabulary,
                'binary': self.bow_binary,
                'lowercase': False,
            }
            if not self.bow_vocabulary:
                del(bow_vectorizer_args['vocabulary'])
            if not self.bow_strip_accents:
                del(bow_vectorizer_args['strip_accents'])
            if not self.bow_stop_words:
                del(bow_vectorizer_args['stop_words'])
            if self.bow_min_df == 1.0:
                # Workaround for defaulting to int as needed for scikit-learn
                bow_vectorizer_args['min_df'] = 1
            bow_vectorizer = BoW_Vectorizer(**bow_vectorizer_args)
            data = self.get_data()
            # Save the BoW representation of the data
            self.engine_object_data = bow_vectorizer.fit_transform(data)
            self.engine_object_vectorizer = bow_vectorizer
            if save:
                self.save()
            return(self.engine_object_vectorizer)

    def get_engine_object_data(self, reconstruct=False, save=True):
        """
        Retrieves / Reconstructs the BoW representation of the data.
        """
        if self.engine_object_data is not None and not reconstruct:
            return(self.engine_object_data)
        else:
            self.get_engine_object_vectorizer(reconstruct=True, save=save)
            return(self.engine_object_data)

    def get_engine_object(self, reconstruct=False, save=True):
        if self.engine_object is not None and not reconstruct:
            return(self.engine_object)
        # Initialize BoW Vectorizer engine object if necessary
        if self.bow_is_enabled:
            self.get_engine_object_vectorizer(reconstruct=reconstruct,
                                              save=True)
        classifier = self.get_classifier().get_engine_object()
        self.engine_object = classifier
        if save:
            self.save()
        return(self.engine_object)

    def perform_inference(self, recalculate=False, save=True):
        if not self.is_inferred or recalculate:
            # No need for running the inference 'engine_meta_iterations' times
            eo = self.get_engine_object(reconstruct=True)
            # -> Get the data
            if self.bow_is_enabled:
                data = self.get_engine_object_data(
                    reconstruct=recalculate, save=save
                )
            else:
                # Use the UTF-8 code point representation
                data = self.get_data(utf8_point_repr=True)
            # -> Get the labels
            labels = self.get_labels()
            # -> Remove Nones if any
            data, labels = self.remove_nones_from_input(data, labels)
            # -> Run the algorithm and store the updated engine object
            self.engine_object = eo.fit(data, labels)
            # -> Rotate metadata
            self.rotate_metadata()
            # -> Perform Cross Validation
            if self.cv_is_enabled:
                self.perform_cross_validation(data=data, labels=labels,
                                              update_metadata=True)
            # -> Update other metadata
            self.metadata["current_inference"]["bow_is_enabled"] = \
                self.bow_is_enabled
            self.metadata["current_inference"]["input_dimensionality"] = \
                np.shape(data)
            self.metadata["current_inference"]["vectorizer_conf"] = \
                self.get_vect_conf_dict()
            self.metadata["current_inference"]["classifier_conf"] = \
                self.get_classifier().get_conf_dict()
            # -> Set as inferred
            self.is_inferred = True
            if save:
                self.engine_object_timestamp = timezone.now()
                self.save()
        return(self.engine_object)

    def predict(self, texts):
        """
        Classifies a list of observations
        """
        if self.is_inferred:
            if self.bow_is_enabled:
                transformed_text = \
                    self.get_engine_object_vectorizer().transform(texts)
            else:
                max_length = max([len(t) for t in self.get_data()])
                transformed_text = \
                    [[ord(character) for character in text.ljust(max_length)]
                     for text in texts][:max_length]
            classifier = self.get_engine_object()
            return(classifier.predict(transformed_text))
        else:
            return(None)

    def perform_cross_validation(self, data=None, labels=None,
                                 update_metadata=False):
        if data is None:
            if self.bow_is_enabled:
                data = self.get_engine_object_data()
            else:
                data = self.get_data(utf8_point_repr=True)
        if labels is None:
            labels = self.get_labels()
        data, labels = self.remove_nones_from_input(data, labels)
        classifier = self.get_engine_object()
        scores = cross_val_score(
            classifier, data, labels,
            cv=self.cv_folds, scoring=self.cv_metric
        )
        if update_metadata:
            self.metadata["current_inference"]['cv'] = {}
            self.metadata["current_inference"]['cv']['conf'] = {
                "folds": self.cv_folds,
                "metric": self.get_cv_metric_display()
            }
            self.metadata["current_inference"]['cv']['scores'] = scores
            self.metadata["current_inference"]['cv']['mean'] = scores.mean()
            self.metadata["current_inference"]['cv']['2std'] = 2 * scores.std()
        return(scores)

    def remove_nones_from_input(self, data, labels):
        # -> Remove data with missing labels if any
        none_indices = [i for i, label in enumerate(labels)
                        if label is None]
        if none_indices:
            if isinstance(data, csr_matrix):
                mask = np.ones(data.shape[0], dtype=bool)
                mask[none_indices] = False
                data = data[mask]
            else:
                data = np.delete(data, none_indices, 0)
            labels = np.delete(labels, none_indices, 0).astype(bool)
        return(data, labels)

    def get_vect_conf_str(self):
        """
        Vectorizer summary configuration string
        """
        vcstr = ""
        if self.bow_is_enabled:
            vcstr += "BoW Representation: "
            if self.bow_binary:
                vcstr += "Binary"
            else:
                if self.bow_use_tf_idf:
                    vcstr += "(TF-IDF Transformation) "
                vcstr += "Analyzer: "
                vcstr += self.get_bow_analyzer_display()
                vcstr += " ({}, {}) - ".format(self.bow_ngram_range_min,
                                               self.bow_ngram_range_max)
                vcstr += "Min / Max DF: "
                vcstr += "{} / {}".format(self.bow_min_df,
                                          self.bow_max_df)
        else:
            vcstr += "UTF-8 Representation (Vectorizer not enabled)"
        return(vcstr)

    def get_vect_conf_dict(self):
        """
        Vectorizer summary configuration string
        """
        vcdict = {}
        vcdict['bow_is_enabled'] = self.bow_is_enabled
        vcdict['bow_use_tf_idf'] = self.bow_use_tf_idf
        vcdict['binary'] = self.bow_binary
        vcdict['analyzer'] = self.get_bow_analyzer_display()
        vcdict['ngram_range'] = "({}, {})".format(self.bow_ngram_range_min,
                                                  self.bow_ngram_range_max)
        vcdict['df_min_max'] = "{} / {}".format(self.bow_min_df,
                                                self.bow_max_df)
        vcdict['str'] = self.get_vect_conf_str()
        return(vcdict)


class IsSpammable(models.Model):
    """
    This Abstract Model (AM) is meant to be used in Django models which may
    recieve Spam.

    Usage:
        - Make your model inherit from this AM.
        - Set the SPAM_FILTER constant to the name of the Spam Filter object
          you would like to use
        - Set the SPAMMABLE_FIELD to the name of the field which stores the
          content.
        - Example::

            class CommentsOfMySite(IsSpammable):
                SPAM_FILTER = "Comment Spam Filter"
                SPAMMABLE_FIELD = "comment"
                ... # The rest of your code
    """
    #: Name of the field which stores the Spammable Content
    SPAMMABLE_FIELD = None
    #: Name of the field which stores the Spam labels
    SPAM_LABEL_FIELD = "is_spam"
    #: Name of the Spam Filter object to be used
    SPAM_FILTER = None

    #: If the object is Spam - Label of the Object
    is_spam = models.NullBooleanField(
        _("Is Spam?"),
        help_text=_((
            'If the object is Spam'
        ))
    )
    #: If the object has been misclassified by the Spam Filter -
    #: useful for some algorithms and for understanding the filter
    is_misclassified = models.BooleanField(
        _("Is Misclassified?"),
        default=False,
        help_text=_((
            'If the object has been misclassified by the Spam Filter'
        ))
    )
    #: If the object classification has been revised by a Human -
    #: Need for proper training and automation
    is_revised = models.BooleanField(
        _("Is Revised?"),
        default=False,
        help_text=_((
            'If the object classification has been revised by a Human'
        ))
    )

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        try:
            spam_filter = SpamFilter.objects.get(name=self.SPAM_FILTER)
        except Exception:
            raise ImproperlyConfigured(_(
                "SPAMMABLE MODEL: "
                "The SPAM_FILTER const reffers to a non-existant object")
            )
        try:
            spammable_field = getattr(self, self.SPAMMABLE_FIELD)
        except Exception:
            raise ImproperlyConfigured(_(
                "SPAMMABLE MODEL: "
                "The SPAMMABLE_FIELD const refers to a non-existant field")
            )
        if spam_filter.is_inferred:
            self.is_spam = spam_filter.predict([spammable_field])
        super(IsSpammable, self).save(*args, **kwargs)


class SpamFilterPreTraining(models.Model):
    """
    Abstract Model for pre-training Spam Filters.
    Subclass this Model for incorporating datasets into the training of
    a Spam Filter (the subclass must be set in the Spam Filter's
    ``pretraining`` field).
    """
    #: Name of the field which stores the Spammable Content
    SPAMMABLE_FIELD = "content"
    #: Name of the field which stores the Spam labels
    SPAM_LABEL_FIELD = "is_spam"

    #: Content
    content = models.TextField(
        _("Content")
    )
    #: Spam label
    is_spam = models.BooleanField(
        _("Is Spam?"),
        default=False
    )

    class Meta:
        abstract = True
        verbose_name = "Spam Filter Pre-Training"
        verbose_name_plural = "Spam Filter Pre-Trainings"

    def __str__(self):
        is_spam = "SPAM" if self.is_spam else "HAM"
        return("[{}] {}...".format(is_spam, self.content[:20]))
