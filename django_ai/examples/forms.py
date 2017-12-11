# -*- coding: utf-8 -*-

from django import forms

from .models import CommentOfMySite


class CommentOfMySiteForm(forms.ModelForm):

    class Meta:
        model = CommentOfMySite
        fields = ('comment', 'user_id', )
        widgets = {
            'user_id': forms.HiddenInput(),
            'comment': forms.Textarea(attrs={'class': 'materialize-textarea'})
        }
