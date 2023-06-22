from django.template import Library
from django.contrib.contenttypes.models import ContentType
from django.urls import reverse


register = Library()


@register.filter
def get_item(dictionary, key):
    return dictionary.get(str(key), "")


@register.filter
def items(dictionary):
    return [(item[0], item[1]) for item in dictionary.items()]


@register.filter
def is_dict(item):
    return isinstance(item, dict)


@register.filter
def is_list(item):
    return isinstance(item, list)


@register.filter
def is_final_dict(dictionary):
    no_dict_or_list_in_values = all(
        [not isinstance(val, (dict, list)) for val in dictionary.values()]
    )
    return no_dict_or_list_in_values


@register.simple_tag
def action_url(action, action_object=None):
    if action_object:
        ct = ContentType.objects.get_for_model(action_object)
        return reverse(
            "run-action",
            kwargs={
                "action": action,
                "content_type": ct.model,
                "object_id": action_object.id,
            },
        )
    else:
        return reverse("run-action", kwargs={"action": action})


@register.inclusion_tag(
    "admin/ai_base/engineobjectmodel/snippets/ai_actions.html",
    takes_context=True,
)
def ai_actions(context):
    return {"original": context["original"]}


@register.inclusion_tag(
    "ai_base/inference_metadata.html",
)
def format_inference_metadata(inference_metadata, descriptions=None):
    return {"metadata": inference_metadata, "descriptions": descriptions}


@register.inclusion_tag(
    "ai_base/metadata_dict.html",
)
def format_metadata_dict(metadata_dict, descriptions=None):
    return {"metadata_dict": metadata_dict, "descriptions": descriptions}
