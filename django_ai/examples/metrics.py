# -*- coding: utf-8 -*-

"""
Metrics to be evaulated on the Metrics Pipeline
-----------------------------------------------

Any function defined or imported here whose name starts with "metric_" will be
executed by the Metrics Pipeline with the data as its first positional argument.

A metric function should pass if the data does not corresponds to what it is
intended to record.
"""

from examples.models import UserInfo


def metric_visits_and_avg_time_page_X(data):
    """
    Updates the average time on pages of type X and its amount of visits
    """
    if data["metric"] == "time_spent" and data["page_type"]:
        ptype = data["page_type"].lower()
        if not ptype in "abcdefghij":
            return(False) # Ignore the data
        ui = UserInfo.objects.get(id=data["user_id"])
        avg_time_pages_X = "avg_time_pages_" + ptype
        visits_pages_X = "visits_pages_" + ptype
        #
        time_spent = float(data["time_spent"])
        n = getattr(ui, visits_pages_X)
        avg_time = getattr(ui, avg_time_pages_X)
        updated_avg_time = (avg_time * (n / (n + 1)) + (time_spent / (n + 1)))
        setattr(ui, avg_time_pages_X, updated_avg_time)
        setattr(ui, visits_pages_X, n + 1)
        ui.save()
        return(True)
    else:
        pass


def metric_visits_and_avg_time_on_pages(data):
    """
    Updates the average time on pages and its amount of visits
    """
    if data["metric"] == "time_spent":
        ui = UserInfo.objects.get(id=data["user_id"])
        time_spent = float(data["time_spent"])
        n = ui.visits_pages
        ui.avg_time_pages = (ui.avg_time_pages * (n / (n + 1)) +
                             (time_spent / (n + 1)))
        ui.visits_pages = n + 1
        ui.save()
        return(True)
    else:
        pass
