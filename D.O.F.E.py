# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:21:25 2021

"""

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import time

from socket import *


# chrome_dr="C:\\Users\\yu065_adadcw1\\Desktop\\parkjeongho\\ai\\chrome_driver\\chromedriver.exe"
os.chdir("D:\\도현\\졸업\\졸작\\작품_진행")
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'face_emotion_model.h5'

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path, compile=False)
print(emotion_classifier.summary())
EMOTIONS = ["angry", "happy", "sad", "surprised", "neutral"]

cv2.namedWindow('your_face')
camera = cv2.VideoCapture(1) # 외부캠 로딩
h = 0
l = 0
emotion_get = []
listent = False

ip = "172.30.10.22"
port = 12345

clientSocket = socket(AF_INET, SOCK_STREAM)  # 소켓 생성
clientSocket.connect((ip, port))  # 서버와 연결

while True:
    listen = False
    if l == 0:
        emotion_get = []
    frame = camera.read()[1]
    # reading the frame
    frame = cv2.resize(frame,(800, 640))  # 이미지를 읽어오는 부분
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 여기서 카메라로 읽은 이미지 grayscale로 변환
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)  # 여기서 얼굴 좌표 반환,      minNeighbors인자의 숫자가 클수록 하나의 얼굴을 정확하게 dstection한다.

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:  # 만약 얼굴이 있다면
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # grayscale image에서 얼굴의 ROI를 추출 후 고정된 28x28 픽셀로 크기를 조정, 준비
        # 분류에 대한 ROI via CNN
        roi = gray[fY:fY + fH, fX:fX + fW]  # 이미지에서 얼굴부분만 자르는 부분
        roi = cv2.resize(roi, (48, 48))  # 얼굴부분만 짜른 이미지를 48x48이미지로 만들어줌
        roi = roi.astype("float") / 255.0  # 픽셀을 0-1사이 숫자로 만듬
        roi = img_to_array(roi)  # 이미지를 어레이로 만들어줌
        roi = np.expand_dims(roi, axis=0)  # (48,48,1)을 (1,48,48,1)로 만듬

        preds = emotion_classifier.predict(roi)[0]  # 여기서 각 class별 예측 확률 뽑음 위 Emotions라는 list와 순서도 동일
        if h == 0 or cv2.waitKey(1) & 0xFF == ord('r'):
            time.sleep(2)

            h += 1
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]  # 여기서 어떤 표정인지 추출

        emotion_get.append(label)

    else:
        continue
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        # draw the label + probability bar on the canvas
        # emoji_face = feelings_faces[np.argmax(preds)]

        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

    #    for c in range(0, 3):
    #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
    #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
    #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
        print("Emotion transported!")
        clientSocket.send(label.encode("utf-8")) #데이터 송신
        data = clientSocket.recv(1024)  # 데이터 수신
        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        print(l)
        if l == 5:
            l = 0
        else:
            l += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
clientSocket.close()
camera.release()
cv2.destroyAllWindows()