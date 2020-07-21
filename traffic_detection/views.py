import json
import os
import shutil
from datetime import time
from io import BytesIO

from PIL import Image as IMG
import cv2

from django.http import JsonResponse, HttpResponse, StreamingHttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect

# Create your views here.
from django.template import loader
from django.views.decorators.csrf import csrf_exempt, csrf_protect

from graduation_project.settings import BASE_DIR
from traffic_detection.darknet import *
from traffic_detection.detect import *
from traffic_detection.forms import *
from traffic_detection.models import Image


@csrf_exempt
def homepage(request):
    if request.method == 'GET':
        return render(request, 'homepage.html')


@csrf_exempt
def imagepage(request):
    if request.method == 'GET':
        form = ImageFileUploadForm();
        deleteAllImage();
        return render(request, 'detect_img.html', {'form': form})


@csrf_exempt
def refresh(request):
    if request.method == 'POST':
        folder = 'D:/graduation_project/media/output'
        lenghtAllFiles = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]);
        if lenghtAllFiles != 0:
            deleteAllImage();
        return JsonResponse({'error': False, 'message': 'Ok'})


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        form = ImageFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            imgage = Image.objects.create();
            image_file = request.FILES['image']
            imgage.image.save(image_file.name, image_file)
            return JsonResponse({'error': False, 'message': 'Uploaded Successfully'})
        else:
            return JsonResponse({'error': True, 'message': 'Some thing wrong'})


def get_latest_file(valid_extensions=('jpg', 'jpeg', 'png', 'mp4'), dirpath='.\\media\\images'):
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
                   f.rsplit('.', 1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)
    return max(valid_files, key=os.path.getmtime)


@csrf_exempt
def detectyolov4(request):
    if request.method == 'POST':
        latest_image = get_latest_file('jpg')
        boxes, path = detect_img(latest_image)

        return JsonResponse({'url': '.\\media\\output\\' + path, 'boxes':boxes})

@csrf_exempt
def detecttraffic(request):
    if request.method == 'POST':
        latest_image = get_latest_file('jpg')
        boxes, path,color_set= detect_img_v1(latest_image)
        return JsonResponse({'url': '.\\media\\output\\' + path, 'boxes': boxes,'color_set':color_set})

@csrf_exempt
def videopage(request):
    if request.method == 'GET':
        form = VideoFileUploadForm();
        deleteAllImage();
        return render(request, 'detect_video.html', {'form': form})


@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        form = VideoFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = Video.objects.create();
            video_file = request.FILES['videofile']
            video.videofile.save(video_file.name, video_file)
            return JsonResponse({'error': False, 'message': 'Uploaded Successfully'})
        else:
            return JsonResponse({'error': True, 'message': 'Some thing wrong'})


# def stream_video():
#     cap = cv2.VideoCapture(get_latest_file('mp4', '.\\media\\videos'))
#     # cap = cv2.VideoCapture(0)
#     cap.set(3, 1280)
#     cap.set(4, 720)
#     out = cv2.VideoWriter(
#         "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))
#     darknet_image = make_image(1280, 720, 3)
#     while True:
#         darknet_image = make_image(1280, 720, 3)
#
#         ret, frame_read = cap.read()
#         frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
#         frame_resized = cv2.resize(frame_rgb,
#                                    (1280, 720),
#                                    interpolation=cv2.INTER_LINEAR)
#
#         copy_image_from_bytes(darknet_image, frame_resized.tobytes())
#
#         detections = detect_vid(darknet_image)
#
#         image = cvDrawBoxes(detections, frame_resized)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         out.write(image)
#         cv2.imwrite('demo.jpg', image)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')
#         cv2.waitKey(1)
#     cap.release()
#     out.release()


def deleteAllImage():
    folder = 'D:/graduation_project/media/output'
    filesToRemove = [os.path.join(folder, f) for f in os.listdir(folder)]
    for f in filesToRemove:
        os.remove(f)


@csrf_exempt
def webcampage(request):
    deleteAllImage();
    return render(request, 'webcam.html');
