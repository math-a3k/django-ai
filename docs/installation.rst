.. _installation:

============
Installation
============

For installing ``django-ai`` in your Django project use the following steps:

1. Activate your virtual environment and then::
    
    pip install django-ai

2. Add it to your `INSTALLED_APPS`::
    
    INSTALLED_APPS = (
        ...
        # Dependencies
        'nested_admin',

        # django-ai apps
        'django_ai.base',
        'django_ai.bayesian_networks',
        'django_ai.supervised_learning',
        'django_ai.systems.spam_filtering',

        # optional but highly recommended
        'django_ai.examples',
        ...
    )

The ``django_ai.examples`` is optional but it is highly recommended that you keep it as a reference.

3. Create the migrations for the dependencies and apply them::
    
    python manage.py makemigrations
    python manage.py migrate

4. Add django-ai's apps URL patterns and its dependencies::
    
    urlpatterns = [
        ...
        url(r'^nested_admin/', # Dependency
            include('nested_admin.urls')),
        url(r'^django-ai/',
            include(django_ai.base.urls)),
        ...
    ]

5. Ensure that the ``admin`` app is enabled.

6. Ensure that your ``static`` serving is properly configured, if not you may have to add to your ``urls.py``::

    ...
    from django.conf.urls.static import static

    urlpatterns = [
        ...
    ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

For reference, there is a working :download:`settings.py <../django_ai/django_ai/settings.py>` and :download:`urls.py <../django_ai/django_ai/urls.py>` in the source distribution for further troubleshooting if necessary.
