import base64
import glob
import json
import cv2
import numpy as np
from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer

from traffic_detection.darknet import *
from traffic_detection.detect import detect_vid
from traffic_detection.views import get_latest_file


def check_device(message):
    if message == 'video':
        return get_latest_file('mp4', '.\\media\\videos')
    elif message == 0:
        return 0
    else:
        return "http://" + message + "/video"


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.asarray([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name

        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )

        self.accept()

    def disconnect(self, close_code):
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        # message = event['message']
        if message == 'break': return
        cap = cv2.VideoCapture(check_device(message))
        cap.set(3, 1280)
        cap.set(4, 720)
        i = 0;
        out = cv2.VideoWriter(
            "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))
        while True:
            if message == 'break': break
            darknet_image = make_image(1280, 720, 3)
            ret, frame_read = cap.read()
            if ret is None:
                break;
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (1280, 720),
                                       interpolation=cv2.INTER_LINEAR)

            copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = detect_vid(darknet_image)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = adjust_gamma(image, 1.5)
            path = './media/output/demo' + str(i) + '.jpg'
            # cv2.imwrite(path,image)
            retval, buffer = cv2.imencode('.jpg', image)
            jpg_as_text = base64.b64encode(buffer)
            self.send(text_data=jpg_as_text.decode('utf-8'))
            # with open(path, "rb") as img_file:
            #     return self.send(text_data=base64.b64encode(image).decode('utf-8'))
            # encoded_string = base64.b64encode(image)
            # res = encoded_string.decode('utf-8')
            #
            # self.send(text_data= res)
            i = i + 1

        out.release()
        cap.release()
        # Send message to room group
        # self.channel_layer.group_send(
        #     self.room_group_name,
        #     {
        #         'type': 'chat_message',
        #         'message': message
        #     })


    # Receive message from room group
    # def chat_message(self, event):
    #
    #     # Send message to WebSocket
