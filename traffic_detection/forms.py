from django import forms

from traffic_detection.models import *


class ImageFileUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('image',)


class VideoFileUploadForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ('videofile',)
