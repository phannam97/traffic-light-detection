"""graduation_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from graduation_project import settings
from traffic_detection.views import *

urlpatterns = [
    path('', homepage),
    path('image-page', imagepage, name='image page'),
    path('video-page', videopage, name='videos page'),
    path('upload_image', upload_image, name="upload_image"),
    path('upload_video', upload_video, name="upload_video"),
    path('detect-yolov4', detectyolov4, name="detect-yolov4"),
    path('webcam-page', webcampage, name="webcam page"),
    path('refresh',refresh),
    path('detect-traffic',detecttraffic)
    # path('admin/', admin.site.urls),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
