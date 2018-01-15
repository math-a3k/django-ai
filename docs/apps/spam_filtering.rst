.. _spam_filtering:

==============
Spam Filtering
==============

This system provides `Spam Filtering <https://en.wikipedia.org/wiki/Spamming>`_ through integrating the *scikit-learn* framework: http://scikit-learn.org.

It provides a pluggable filter for any Django model that is subject to Text Spam.

An example of implementing a Spam Filter into a project can be found in :ref:`example_spam_filtering`.

.. _spam_filter:

Spam Filter
===========

This is the main object for the Spam Filtering System.

.. autoclass:: systems.spam_filtering.models.SpamFilter

All the configuration can be done through the admin of Spam Filters - or more specifically, through the `change form`.

Front-End
---------

General
^^^^^^^
General fields (like ``Name``) and Miscellanous are documented in the :ref:`api_statistical_model`.

The implementation uses *scikit-learn* as Engine, there is no need of setting more than 1 ``Engine Meta Iterations``.


Spammable Model
^^^^^^^^^^^^^^^
A Spammable Model is a Django model which inherits from the :ref:`is_spammable` Abstract Model (discussed below) for convenience of incorparting the model to all the functionality in the Spam Filtering cycle.

.. autoattribute:: systems.spam_filtering.models.SpamFilter.spam_model_is_enabled
    :annotation: Use a Spammable Model?

.. autoattribute:: systems.spam_filtering.models.SpamFilter.spam_model_model
    :annotation: Spammable Django Model

If you choose not to use an Spammable Model, you can specify where the data is held (Spammable Content and Labels) via the Data Columns and Labels Column sections.

Classifier
^^^^^^^^^^
The Classifier model to be used for discerning the Spam.

Any implementation of a :ref:`api_supervised_learning` using a `scikit-learn` classifier will work.

.. autoattribute:: systems.spam_filtering.models.SpamFilter.classifier
    :annotation: Classifier to be used in the System

Cross Validation
^^^^^^^^^^^^^^^^
`Cross Validation (CV) <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ will be used as the perfomance estimation of the Spam Filter. The reported estimation will be the mean and the 2 standard deviations interval of the metrics evaluated in each CV fold.

CV is done with the `scikit-learn` engine, more general information is available `here <http://scikit-learn.org/stable/modules/cross_validation.html>`_ and `here is detailed about the available metrics <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_. 

.. autoattribute:: systems.spam_filtering.models.SpamFilter.cv_is_enabled
    :annotation: Enable Cross Validation?

.. autoattribute:: systems.spam_filtering.models.SpamFilter.cv_folds
    :annotation: Cross Validation Folds

.. autoattribute:: systems.spam_filtering.models.SpamFilter.cv_metric
    :annotation: Cross Validation Metric

Pre-Training
^^^^^^^^^^^^

Pre-training refers to providing the model with "initial" data, as "initializating" the model. See :ref:`spam_filter_pre_training` for more details.

.. autoattribute:: systems.spam_filtering.models.SpamFilter.pretraining
    :annotation: 

Bag of Words Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Bag of Words representation (BoW) <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ is a suitable representation for many Natural Language Processing problems - such as text classification.

If it is not enabled, the Spam Filter will use the `UTF-8 code point representation` for the corpus: each character is represented on an axis and its value is its UTF-8 code, i.e. ``Hola!HOLA!`` will be represented as ( 72, 111, 108,  97,  33,  72,  79,  76,  65,  33), and the input dimensionality will be the maximum length of the texts in the corpus.

For more information on the transformation, see the :ref:`example_spam_filtering` and the `Engine documentation <http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction>`_.

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_is_enabled
    :annotation: Enable Bag of Words representation?

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_use_tf_idf
    :annotation: (BoW) Use TF-IDF transformation?

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_analyzer
    :annotation: (BoW) Analyzer

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_ngram_range_min
    :annotation: (BoW) n-gram Range - Min

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_ngram_range_max
    :annotation: (BoW) n-gram Range - Max

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_max_df
    :annotation: (BoW) Maximum Document Frequency

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_min_df
    :annotation: (BoW) Minimum Document Frequency

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_max_features
    :annotation: (BoW) Maximum Features

Bag of Words Transformation - Miscellanous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_binary
    :annotation: (BoW) Use Binary representation?

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_enconding
    :annotation: (BoW) Encoding

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_decode_error
    :annotation: (BoW) Decode Error

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_strip_accents
    :annotation: (BoW) Strip Accents

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_stop_words
    :annotation: (BoW) Stop Words

.. autoattribute:: systems.spam_filtering.models.SpamFilter.bow_vocabulary
    :annotation: (BoW) Vocabulary

API
---

`SpamFilter` extends the :ref:`api_supervised_learning` in several ways.

Engine-related
^^^^^^^^^^^^^^

The Engine's Vectorizer is available / stored in a `PickleField`:

.. autoattribute:: systems.spam_filtering.models.SpamFilter.engine_object_vectorizer

and initialized / retrieved with:

.. automethod:: systems.spam_filtering.models.SpamFilter.get_engine_object_vectorizer

returning an instance of `CountVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer>`_ or `TdifVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer>`_ according to the :attr:`Spam Filter's configuration <systems.spam_filtering.models.SpamFilter.bow_use_tf_idf>`.

The BoW-transformed data is available / stored also in a `PickleField`:

.. autoattribute:: systems.spam_filtering.models.SpamFilter.engine_object_data

which can be retrieved or reconstructed via

.. automethod:: systems.spam_filtering.models.SpamFilter.get_engine_object_data

Classification is done by the ``predict`` method - as expected - but with the caveat of taking a list of observations (texts) as its argument:

.. automethod:: systems.spam_filtering.models.SpamFilter.predict

.. code-block:: python

    from django_aisystems.spam_filtering.models import SpamFilter

    sf = SpamFilter.objects.get(name="<NAME-OF-MY-SF>")
    sf.predict(["Buy Viagra Online"])
    sf.predict(["Buy Cialis Online", "Oh, loved your article!"])

.. _is_spammable:

IsSpammable
===========

`IsSpammable` is a Django Abstract Model (AM) meant to give convenience in the Spam Filtering cycle.

The AM provides the fields, options and ``.save()`` method to attach the model to a Spam Filter.

Once attached to a Spam Filter, the data held in the Django model will be used for training the Filter and the Filter will be used to classify new data created in the model (on ``.save()`` if the Spam Filter is inferred).

.. autoclass:: systems.spam_filtering.models.IsSpammable

Fields and Settings
-------------------

.. autoattribute:: systems.spam_filtering.models.IsSpammable.SPAMMABLE_FIELD
.. autoattribute:: systems.spam_filtering.models.IsSpammable.SPAM_LABEL_FIELD
.. autoattribute:: systems.spam_filtering.models.IsSpammable.SPAM_FILTER
.. autoattribute:: systems.spam_filtering.models.IsSpammable.is_spam
    :annotation: Is Spam?
.. autoattribute:: systems.spam_filtering.models.IsSpammable.is_misclassified
    :annotation: Is Misclassified?
.. autoattribute:: systems.spam_filtering.models.IsSpammable.is_revised
    :annotation: Is Revised?

Usage
-----

- Make your model inherit from this AM.
- Choose the Spam Filter to be attached by seting the ``SPAM_FILTER`` constant to the name of the Spam Filter object.
  you would like to use
- Set the ``SPAMMABLE_FIELD`` constant to the name of the field which stores the content.
- Make and run migrations.

Example
-------

.. code-block:: python

    class CommentsOfMySite(IsSpammable):
        SPAM_FILTER = "Comment Spam Filter"
        SPAMMABLE_FIELD = "comment"
        ... # The rest of your code

Other Considerations
--------------------

Technically, what makes a Django model "pluggable" into a Spam Filter as a source of data for training are:

- ``SPAMMABLE_FIELD`` constant which defines the where is the content
- ``SPAM_LABEL_FIELD`` constant which defines the field where the label is stored - defaulted to ``ìs_spam``.
- A `NullBooleanField` to store the labels of the objects.
  
If you do not want ot inherit from the AM, any model with these three defined will work as an Spammable Model in the Spam Filter setup. The only pending thing for completing the systmes is the automation of classification of new objects.  

.. _spam_filter_pre_training:

Spam Filter Pre-Training
========================

Pre-training refers to providing the model with other data, "external" data, as an initialization. That data is incorporated into the training dataset of the model.

`SpamFilterPreTraining` is a Django Abstract Model (AM) meant to give convenience in pre-training the Spam Filter.

.. autoclass:: systems.spam_filtering.models.SpamFilterPreTraining

Usage
-----

- Create a Django Model that inherits from `SpamFilterPreTraining`
- Make and run migrations
- Import data to the Django Model
- Set the Spam Filter pre-training field to use the pre-training model

Example
-------

.. code-block:: python

    class SFPTEnron(SpamFilterPreTraining):

        class Meta:
            verbose_name = "Spam Filter Pre-Training: Enron Email Data"
            verbose_name_plural = "Spam Filter Pre-Training: Enron Emails Data"

.. autofunction:: examples.migrations.0015_sfptenron_sfptyoutube.download_and_process_pretrain_data_files

Other Considerations
--------------------

Technically, what makes a Django model "pluggable" into a Spam Filter as a source of pre-training are the ``content`` and ``is_spam`` fields, or the ``SPAMMABLE_FIELD`` and ``SPAM_LABEL_FIELD`` constants defined in the class pointing to Text or Char field and a Boolean field respectively.

If you do not want to inherit, define either or both in your Django Model and it will be "pluggable" as a pre-training dataset.
