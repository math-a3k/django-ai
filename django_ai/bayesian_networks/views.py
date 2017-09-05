# -*- coding: utf-8 -*-

from django.shortcuts import (redirect, get_object_or_404)
from django.contrib.auth.decorators import user_passes_test

from .models import BayesianNetwork


@user_passes_test(lambda u: u.is_superuser)
def bn_run_inference(request, bn_id):
    bn = get_object_or_404(BayesianNetwork, pk=bn_id)
    bn.perform_inference(recalculate=True)

    return redirect(request.META.get('HTTP_REFERER', '/'))


@user_passes_test(lambda u: u.is_superuser)
def bn_reset_inference(request, bn_id):
    bn = get_object_or_404(BayesianNetwork, pk=bn_id)
    bn.reset_inference()

    return redirect(request.META.get('HTTP_REFERER', '/'))
