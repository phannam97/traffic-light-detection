from django.db import models

# Create your models here.
from django.db import models


# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to='images/')


class Video(models.Model):
    videofile= models.FileField(upload_to='videos/', null=True, verbose_name="")