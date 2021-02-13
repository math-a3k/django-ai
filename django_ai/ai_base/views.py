import inspect
import random
import numpy as np

from django.contrib import messages
from django.views.generic import RedirectView
from django.contrib.contenttypes.models import ContentType
from django.http import Http404
from django.contrib.auth.mixins import UserPassesTestMixin


class RunActionView(UserPassesTestMixin, RedirectView):
    """
    Runs common Actions for Systems and Techniques
    """
    permanent = False
    #: Available Actions
    ACTIONS = {
        "perform_inference": {
            "type": "object",
            "str": "PERFORMING INFERENCE",
            "method": "perform_inference",
            "kwargs": {},
        },
        "reset_inference": {
            "type": "object",
            "str": "RESETING INFERENCE",
            "method": "reset_inference",
            "kwargs": {},
        },
        "reset_metadata": {
            "type": "object",
            "str": "RESETING METADATA",
            "method": "reset_metadata",
            "kwargs": {},
        },
        "reinitialize_rng": {
            "type": "general",
            "str": "REINITIALIZING RNG",
            "method": "action_reinitialize_rng",
            "kwargs": {},
        }
    }

    def test_func(self):
        return self.request.user.is_superuser or self.request.user.is_staff

    def action_reinitialize_rng(self):
        """
        Reinitialize both generators
        """
        random.seed()
        np.random.seed()

    def get_ct_object(self, content_type, object_id):
        ct = ContentType.objects.get(model=content_type)
        return ct.model_class().objects.get(id=object_id)

    def run_action(self, action, action_object=None):
        try:
            if action_object:
                action_method = getattr(action_object, action['method'])
            else:
                action_method = getattr(self, action['method'])
            action_method(**action['kwargs'])
            messages.success(self.request,
                             "SUCCESS AT {}".format(action['str']))
        except Exception as e:
            msg = e.args[0]
            frm = inspect.trace()[-1]
            mod = inspect.getmodule(frm[0])
            modname = mod.__name__ if mod else frm[1]
            messages.error(self.request,
                           "ERROR WHILE {}: [{}] {}".format(
                               action['str'], modname, str(msg)))

    def get_redirect_url(self, *args, **kwargs):
        if kwargs['action'] not in self.ACTIONS:
            raise Http404("Action not Found")
        if self.ACTIONS[kwargs['action']]["type"] == 'object':
            action_object = self.get_ct_object(kwargs['content_type'],
                                               kwargs['object_id'])
        else:
            action_object = None
        self.run_action(self.ACTIONS[kwargs['action']], action_object)
        return self.request.META.get('HTTP_REFERER', '/')
