# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import

from django.conf.urls import (url, include)
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^bayesian_networks/', include('django_ai.bayesian_networks.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
