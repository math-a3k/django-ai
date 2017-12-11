# -*- coding: utf-8 -*-

import random
import math
import json

from django.shortcuts import (render, redirect, )
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Avg
from django.views.generic import CreateView
from django.urls import reverse_lazy

from bayesian_networks.models import BayesianNetwork
from systems.spam_filtering.models import SpamFilter
from examples import metrics
from .models import (UserInfo, CommentOfMySite, )
from .forms import (CommentOfMySiteForm, )

PAGES_COLORS = {
    "A": "green", "B": "deep-orange", "C": "blue-grey", "D": "brown",
    "E": "lime", "F": "light-blue", "G": "teal", "H": "deep-purple",
    "I": "indigo", "J": "red"
}

USERS = getattr(settings, "DJANGO_AI_EXAMPLES_USERINFO_SIZE", 200)


def another_page(besides):
    other_pages = list(set(PAGES_COLORS.keys()) - set(besides))
    return(random.choice(other_pages))


def a_page_of_type_X(request, page_type="A", user_id=random.randint(1, USERS)):
    """
    View that renders pages of type X.
    user_id is for mimicking the user logged in (request.user).
    """
    bn = BayesianNetwork.objects.get(name="Clustering (Example)")
    if not bn.is_inferred:
        return(
            render(
                request,    
                template_name="examples/sample_page.html",
                context={"is_inferred": False}
            )
        )
    current_page = page_type
    avg_time_pages_X = "avg_time_pages_" + current_page.lower()
    user_info = UserInfo.objects.get(id=user_id)
    users_ids = UserInfo.objects.all().values_list("id", flat=True)\
        .order_by('-id')[:200]  # Prevents the browser becoming unresponsive
    cluster_avg_time_pages = \
        bn.metadata["clusters_means"][user_info.cluster_1][0]
    # Calculate the cluster mean for the avg time of this type of page.
    # It may differ in the case of type A with the cluster mean of bn.metadata
    # calculated at the moment of performing the inference
    cluster_avg_times = UserInfo.objects\
        .filter(cluster_1=user_info.cluster_1)\
        .aggregate(**{avg_time_pages_X: Avg(avg_time_pages_X)})
    #
    return(
        render(
            request,
            template_name="examples/sample_page.html",
            context={
                "is_inferred": True,
                "page_type": page_type,
                "page_color": PAGES_COLORS[page_type],
                "current_user": int(user_id),
                "users_ids": users_ids,
                "other_page": another_page(besides=current_page),
                "cluster": user_info.cluster_1,
                "user_avg_time_pages_of_this_type":
                    getattr(user_info, avg_time_pages_X),
                "cluster_avg_time_pages_of_this_type":
                    cluster_avg_times[avg_time_pages_X],
                "user_avg_time_pages": user_info.avg_time_pages,
                "cluster_avg_time_pages": cluster_avg_time_pages,
            }
        )
    )


def new_user(request):
    """
    Mimmics creating a user by creating an UserInfo object.
    """
    bn = BayesianNetwork.objects.get(name="Clustering (Example)")
    new_user = UserInfo.objects.create(
        sex=random.choice([0, 1]),
        age=math.floor(random.gauss(30, 2)),
        avg1=random.gauss(10, 5),
        cluster_1=bn.assign_cluster([0, 0])
    )
    # -> Increment the BN internal counter by 5
    # You **SHOULD NOT** do this, as it will trigger eventually an inference
    # on the model inside of the user's "navigation" request cycle (if
    # BN.counter_threshold is set)
    bn.counter += 5
    bn.save()
    # -> Instead, update the counter "directly" in the database:
    # BayesianNetwork.objects.filter(name="Clustering (Example)")\
    #     .update(counter=F("counter") + 5)
    # and / or schedule a model recalculation.
    return(redirect('page', page_type="A", user_id=new_user.id))


METRICS_PIPELINE = {
    m: metrics.__dict__[m]
    for m in metrics.__dict__ if m.startswith("metric")
}


@csrf_exempt
def process_metrics(request, verbose=True):
    """
    Minimal implementation of a Metricps Pipeline.

    The CSRF exemption is because it is highly unlikely that an external site
    posts values to our localhost to mess with our AI :)
    """
    if request.method == "POST":
        data = json.loads(request.body.decode())
        if verbose:
            print(data, METRICS_PIPELINE)
        for metric_name, metric in METRICS_PIPELINE.items():
            metric(data)
        # -> Increment the BN internal counter by 1
        # No problem doing this, as it is outside of the user's "navigation"
        # request cycle:
        bn = BayesianNetwork.objects.get(name="Clustering (Example)")
        bn.counter += 1
        bn.save()
        # You could also update BN internal counter "directly",
        # without .save():
        # BayesianNetwork.objects.filter(name="Clustering (Example)")\
        #     .update(counter=F("counter") + 1)
        # and / or schedule a model recalculation.
        return(HttpResponse(status=204))
    else:
        return(HttpResponse(status=400))


class MetricsMixin(object):

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Usage metrics
        bn = BayesianNetwork.objects.get(name="Clustering (Example)")
        user_info = UserInfo.objects.get(id=self.user_id)
        cluster_avg_time_pages = \
            bn.metadata["clusters_means"][user_info.cluster_1][0]
        context['cluster_avg_time_pages'] = cluster_avg_time_pages
        return context


class CommentsOfMySiteView(MetricsMixin, CreateView):
    template_name = "examples/comments_of_my_site.html"
    form_class = CommentOfMySiteForm
    user_id = None

    def get_user_id(self):
        self.user_id = int(self.kwargs.get('user_id',
                                           random.randint(1, USERS)))
        return(self.user_id)

    def get_success_url(self):
        return(reverse_lazy('comments-of-my-site',
                            kwargs={'user_id': self.get_user_id()}))

    def get_initial(self):
        return({'user_id': self.get_user_id()})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['is_inferred'] = True
        context['page_type'] = "C"
        context['page_color'] = PAGES_COLORS["C"]
        context['current_user'] = self.get_user_id()
        users_ids = UserInfo.objects.all().values_list("id", flat=True)\
            .order_by('-id')[:200]  # Prevents becoming unresponsive
        context['users_ids'] = users_ids
        context['users_dropdown_target'] = "comments-of-my-site"
        context['latest_comments'] = \
            CommentOfMySite.objects.all().order_by("-id")[:10]
        context['spam_filter'] = SpamFilter.objects.get(
            name=CommentOfMySite.SPAM_FILTER)
        return context
