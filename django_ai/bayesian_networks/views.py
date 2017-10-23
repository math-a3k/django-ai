# -*- coding: utf-8 -*-

import inspect
import random
import numpy as np

from django.shortcuts import (redirect, get_object_or_404)
from django.contrib.auth.decorators import user_passes_test
from django.contrib import messages

from .models import BayesianNetwork


@user_passes_test(lambda u: u.is_superuser)
def bn_run_inference(request, bn_id):
    bn = get_object_or_404(BayesianNetwork, pk=bn_id)
    try:
        bn.perform_inference(recalculate=True)
    except Exception as e:
        msg = e.args[0]
        frm = inspect.trace()[-1]
        mod = inspect.getmodule(frm[0])
        modname = mod.__name__ if mod else frm[1]
        messages.error(request,
                       "ERROR WHILE PERFORMING INFERENCE: [{}] {}".format(
                           modname, str(msg)))

    return redirect(request.META.get('HTTP_REFERER', '/'))


@user_passes_test(lambda u: u.is_superuser)
def bn_reset_inference(request, bn_id):
    bn = get_object_or_404(BayesianNetwork, pk=bn_id)
    bn.reset_inference()

    return redirect(request.META.get('HTTP_REFERER', '/'))


@user_passes_test(lambda u: u.is_superuser)
def bn_reinitialize_rng(request):
    random.seed()
    np.random.seed()

    return redirect(request.META.get('HTTP_REFERER', '/'))
