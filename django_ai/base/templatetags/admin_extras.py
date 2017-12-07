# -*- coding: utf-8 -*-
from django.template import Library


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
