# -*- coding: utf-8 -*-

from django.template import Library
from django.contrib.contenttypes.models import ContentType
from django.urls import reverse


register = Library()


@register.simple_tag(takes_context=True)
def get_fieldsets_and_inlines(context):
    """
    https://github.com/dezede/dezede/blob/master/libretto/
            templatetags/admin_extras.py
    """
    adminform = context['adminform']
    model_admin = adminform.model_admin
    adminform = list(adminform)
    inlines = list(context['inline_admin_formsets'])

    fieldsets_and_inlines = []
    for choice in getattr(model_admin, 'fieldsets_and_inlines_order', ()):
        if choice == 'f':
            if adminform:
                fieldsets_and_inlines.append(('f', adminform.pop(0)))
        elif choice == 'i':
            if inlines:
                fieldsets_and_inlines.append(('i', inlines.pop(0)))

    for fieldset in adminform:
        fieldsets_and_inlines.append(('f', fieldset))
    for inline in inlines:
        fieldsets_and_inlines.append(('i', inline))

    return(fieldsets_and_inlines)


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)


@register.simple_tag
def action_url(action, action_object=None):
    # import ipdb; ipdb.set_trace()
    if action_object:
        ct = ContentType.objects.get_for_model(action_object)
        return(
            reverse('run-action', kwargs={
                "action": action, "content_type": ct.model,
                "object_id": action_object.id}
            )
        )
    else:
        return(
            reverse('run-action', kwargs={"action": action})
        )


@register.inclusion_tag('ai_base/snippets/ai_actions.html', takes_context=True)
def ai_actions(context):
    return({"original": context['original']})
