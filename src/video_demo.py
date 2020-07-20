
import os
import cv2
import dlib
import time
import imutils
import numpy as np
import argparse
import sys
from keras.models import load_model
from keras.utils.data_utils import get_file
from imutils.video import VideoStream
from age_gender_model import WideResNet

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_path", required = True, help = "path to mp4 video")
ap.add_argument("-e", "--emotion_model", type = str, default = 'emotion.hdf5', help = "path to trained emotion model")
ap.add_argument("-ag", "--age_gender_model", type = str, default = "weights.28-3.73.hdf5", help = "path to trained age-gender model")
args = vars(ap.parse_args())

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

margin = 0.4
img_size = 64
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_model_path = args['emotion_model']
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

pretrained_model = args['age_gender_model']
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

weight_file = get_file(args['age_gender_model'], pretrained_model, cache_subdir = "pretrained_models", file_hash = modhash)
age_gender_model = WideResNet(img_size, depth=16, k=8)()
age_gender_model.load_weights(weight_file)

detector = dlib.get_frontal_face_detector()
vs = cv2.VideoCapture(args['video_path'])

while (vs.isOpened()):
    ret, frame = vs.read()
    if ret == True:
        frame = imutils.resize(frame, width = 600)
        img_h, img_w, _ = np.shape(frame)
        detected = detector(frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (64, 64))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = np.expand_dims(face, 0)
                face = preprocess_input(face)
                face = np.expand_dims(face, -1)
                emotion_label_arg = np.argmax(emotion_classifier.predict(face))
                emotion_text = emotion_labels[emotion_label_arg]
                print(emotion_text)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))
                cv2.putText(frame, emotion_text, (x1,y2) ,cv2.FONT_HERSHEY_SIMPLEX, 0.4 ,(255, 255, 255),  lineType=cv2.LINE_AA)

            results = age_gender_model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                cv2.putText(frame, label, (d.left(), d.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)


        cv2.imshow('disp', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
