# -*- coding: utf-8 -*-

from itertools import (chain, )
from pickle import HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL

from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import ValidationError
from django.contrib.contenttypes.models import ContentType

from picklefield.fields import PickledObjectField
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.model_selection import (cross_val_score, GridSearchCV, )
from sklearn.pipeline import Pipeline

from base.models import (SupervisedLearningTechnique, )
from base.utils import (get_model, )


class SpamFilter(SupervisedLearningTechnique):
    """
    Main object for the Spam Filtering System.
    """
    BOW_DECODE_ERROR_CHOICES = (
        ('strict', _('Strict')),
        ('ignore', _('Ignore')),
        ('replace', _('Replace')),
    )
    BOW_STRIP_ACCENTS_CHOICES = (
        ('ascii', 'ASCII'),
        ('unicode', 'Unicode'),
    )
    BOW_ANALYZER_CHOICES = (
        ('word', _('Word')),
        ('char', _('Character')),
        ('char_wb', _("Characters in Word-Boundaries")),
    )

    classifier = models.CharField(
        "Supervised Learning Classifier",
        max_length=100, blank=True, null=True,
        help_text=(
            'Classifier to be used in the System, in the '
            '"app_label.model|name" format, i.e. '
            '"supervised_learning.SVC|My SVM"'
        )
    )
    engine_object_vectorizer = PickledObjectField(
        "Engine Object Vectorizer",
        protocol=pickle_HIGHEST_PROTOCOL,
        blank=True, null=True
    )
    engine_object_data = PickledObjectField(
        "Engine Object Data",
        protocol=pickle_HIGHEST_PROTOCOL,
        blank=True, null=True
    )
    pretraining = models.CharField(
        "Pre-Training dataset",
        max_length=100, blank=True, null=True,
        help_text=(
            'Django Model containing the pre-training dataset in the'
            '"app_label.model" format, i.e. "examples.SFPTEnron"'
        )
    )

    # encoding : string, ‘utf-8’ by default.
    bow_enconding = models.CharField(
        "BoW encoding",
        default='utf-8', max_length=20,
        help_text=(
            'Encoding to be used to decode.'
        )
    )
    # decode_error : {‘strict’, ‘ignore’, ‘replace’}
    bow_decode_error = models.CharField(
        "BoW Decode Error",
        default='strict', max_length=20, choices=BOW_DECODE_ERROR_CHOICES,
        help_text=_((
            'Instruction on what to do if a byte sequence is given to '
            'analyze that contains characters not of the given encoding. '
            'By default, it is ‘strict’, meaning that a UnicodeDecodeError '
            'will be raised. Other values are ‘ignore’ and ‘replace’.'
        ))
    )
    # strip_accents : {‘ascii’, ‘unicode’, None}
    bow_strip_accents = models.CharField(
        "BoW Strip Accents",
        default=None, max_length=20, choices=BOW_STRIP_ACCENTS_CHOICES,
        blank=True, null=True,
        help_text=_((
            'Remove accents during the preprocessing step. ‘ascii’ is a fast '
            'method that only works on characters that have an direct ASCII '
            'mapping. ‘unicode’ is a slightly slower method that works on '
            'any characters. None (default) does nothing.'
        ))
    )
    # analyzer : string, {‘word’, ‘char’, ‘char_wb’} or callable
    bow_analyzer = models.CharField(
        "BoW Analyzer",
        default='word', max_length=20, choices=BOW_ANALYZER_CHOICES,
        help_text=_((
            'Whether the feature should be made of word or character n-grams. '
            'Option ‘char_wb’ creates character n-grams only from text inside '
            'word boundaries; n-grams at the edges of words are padded with'
            ' space.'
        ))
    )
    # ngram_range : tuple (min_n, max_n)
    bow_ngram_range_min = models.SmallIntegerField(
        "BoW n-gram Range - Min",
        default=1,
        help_text=_((
            'The lower boundary of the range of n-values for '
            'different n-grams to be extracted (comma-separated). All values '
            'of n such that min_n <= n <= max_n will be used.'
        ))
    )
    bow_ngram_range_max = models.SmallIntegerField(
        "BoW n-gram Range - Max",
        default=1,
        help_text=_((
            'The upper boundary of the range of n-values for '
            'different n-grams to be extracted (comma-separated). All values '
            'of n such that min_n <= n <= max_n will be used.'
        ))
    )
    # stop_words : string {‘english’}, list, or None (default)
    bow_stop_words = models.TextField(
        "BoW Stop Words",
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
    # max_df : float in range [0.0, 1.0] or int, default=1.0
    bow_max_df = models.FloatField(
        "BoW Maximum Document Frequency",
        default=1.0,
        help_text=_((
            'When building the vocabulary ignore terms that have a document '
            'frequency strictly higher than the given threshold '
            '(corpus-specific stop words). If float, the parameter represents '
            'a proportion of documents, integer absolute counts. This '
            'parameter is ignored if vocabulary is not None.'
        ))
    )
    # min_df : float in range [0.0, 1.0] or int, default=1
    bow_min_df = models.FloatField(
        "BoW Minimum Document Frequency",
        default=1,
        help_text=_((
            'When building the vocabulary ignore terms that have a document '
            'frequency strictly lower than the given threshold. This value is '
            'also called cut-off in the literature. If float, the parameter '
            'represents a proportion of documents, integer absolute counts. '
            'This parameter is ignored if vocabulary is not None.'
        ))
    )
    # max_features : int or None, default=None
    bow_max_features = models.IntegerField(
        "BoW Maximum Features",
        default=None, blank=True, null=True,
        help_text=_((
            'If not None, build a vocabulary that only consider the top '
            'max_features ordered by term frequency across the corpus. '
            ' This parameter is ignored if vocabulary is not None.'
        ))
    )
    # vocabulary : Mapping or iterable, optional
    bow_vocabulary = models.TextField(
        "BoW Vocabulary",
        default=None, blank=True, null=True,
        help_text=_((
            'Either a Mapping (e.g., a dict) where keys are terms and values '
            'are indices in the feature matrix, or an iterable over terms. '
            'If not given, a vocabulary is determined from the input '
            'documents. Indices in the mapping should not be repeated and '
            'should not have any gap between 0 and the largest index.'
        ))
    )
    # binary : boolean, default=False
    bow_binary = models.BooleanField(
        "BoW Binary",
        default=False,
        help_text=_((
            'If True, all non zero counts are set to 1. This is useful for '
            'discrete probabilistic models that model binary events rather '
            'than integer counts.'
        ))
    )
    # use tf-idf transformation
    bow_use_tf_idf = models.BooleanField(
        "BoW Binary",
        default=False,
        help_text=_((
            'If True, all non zero counts are set to 1. This is useful for '
            'discrete probabilistic models that model binary events rather '
            'than integer counts.'
        ))
    )

    class Meta:
        verbose_name = "Spam Filter"
        verbose_name_plural = "Spam Filters"

    def __str__(self):
        return("[Spam Filter]{}".format(self.name))

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

        super(SpamFilter, self).clean()

    def get_data(self):
        data = super(SpamFilter, self).get_data()
        # Flatten list
        return(list(chain.from_iterable(data)))

    def get_classifier(self):
        app_model, object_name = self.classifier.split("|")
        app, model = app_model.split(".")
        model_class = ContentType.objects.get(
            app_label=app,
            model=model.lower()
        ).model_class()
        return(model_class.objects.get(name=object_name))

    def get_engine_object_vectorizer(self, reconstruct=False, save=True):
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
            if self.pretraining:
                data += list(self.get_pretraining_data())
            # Save the BoW representation of the data
            self.engine_object_data = bow_vectorizer.fit_transform(data)
            self.engine_object_vectorizer = bow_vectorizer
            if save:
                self.save()
            return(self.engine_object_vectorizer)

    def get_engine_object_data(self, reconstruct=False, save=True):
        if self.engine_object_data is not None and not reconstruct:
            return(self.engine_object_data)
        else:
            self.get_engine_object_vectorizer(reconstruct=True, save=save)
            return(self.engine_object_data)

    def get_engine_object(self, reconstruct=False, save=True):
        if self.engine_object is not None and not reconstruct:
            return(self.engine_object)
        classifier = self.get_classifier().get_engine_object()
        bow_data = self.get_engine_object_data(reconstruct=reconstruct,
                                               save=save)
        labels = self.get_labels()
        if self.pretraining:
            labels = list(labels)
            labels += list(self.get_pretraining_labels())
        self.engine_object = classifier.fit(bow_data, labels)
        if save:
            self.save()
        return(self.engine_object)

    def get_pretraining_data(self):
        if self.pretraining:
            model = get_model(self.pretraining)
            return(model.objects.values_list("content", flat=True))
        else:
            return(None)

    def get_pretraining_labels(self):
        if self.pretraining:
            model = get_model(self.pretraining)
            return(model.objects.values_list("is_spam", flat=True))
        else:
            return(None)

    def predict(self, text):
        bow_text = self.get_engine_object_vectorizer().transform(text)
        classifier = self.get_engine_object()
        return(classifier.predict(bow_text))


class IsSpammable(models.Model):
    """
    This Abstract Model (AM) is meant to be used in Django models which may
    recieve Spam.

    Usage
    =====
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
    is_spam = models.BooleanField(
        _("Is Spam?"),
        default=False,
        help_text=_((
            'If the object is Spam'
        ))
    )
    is_misclassified = models.BooleanField(
        _("Is Misclassified?"),
        default=False,
        help_text=_((
            'If the object has been misclassified by the Spam Filter'
        ))
    )
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
        spam_filter = SpamFilter.objects.get(name=self.SPAM_FILTER)
        spammable_field = getattr(self, self.SPAMMABLE_FIELD)
        self.is_spam = spam_filter.predict([spammable_field])
        super(IsSpammable, self).save(*args, **kwargs)


class SpamFilterPreTraining(models.Model):
    """
    Abstract Model for pre-training Spam Filters.
    Subclass this Model for incorporating datasets into the training of
    a Spam Filter (the subclass must be set in the Spam Filter's
    ``pretraining`` field)
    """
    content = models.TextField(
        _("Content")
    )
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
