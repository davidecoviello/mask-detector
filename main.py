import cv2
import argparse
import imutils
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from imutils.video import VideoStream

MODEL_DIR = 'model/'
IMG_HEIGHT = 256
IMG_WIDTH = 256
CLASS_NAMES = ['correctly-masked', 'not-masked', 'incorrectly-masked']
CORRECTLY_MASKED = CLASS_NAMES[0]
NOT_MASKED = CLASS_NAMES[1]
INCORRECTLY_MASKED = CLASS_NAMES[2]

model = keras.models.load_model(MODEL_DIR)

def get_face(img, x, y, w, h, width, height) -> Image.Image: 
    crop_img = img[y:y+h, x:x+w]
    cv2.imshow("cropped", crop_img)
    rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    newsize = (width, height)
    resized_img = pil_img.resize(newsize)
    return resized_img


def predict(resized_img, class_names):
    img_array = keras.preprocessing.image.img_to_array(resized_img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return prediction, confidence

    
def get_color(prediction):
    green = (36,255,12)
    red = (0,0,255)
    yellow = (0,255,255)
    if prediction == CORRECTLY_MASKED:
        color = green
    elif prediction == NOT_MASKED:
        color = red
    else:
        color = yellow

    return color


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("[INFO] starting video stream ...")
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
_, img = cap.read()
if (cap.isOpened() == False):
    print("[ERROR] Unable to read camera feed ...")

while True:
    _, img = cap.read()
    # Convert to grayscale (haarcascade works with grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        resized_img = get_face(img, x, y, w, h, IMG_WIDTH, IMG_HEIGHT)
        prediction, confidence= predict(resized_img, CLASS_NAMES)
        color = get_color(prediction)
        print(f'{prediction}: {confidence}')
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, str(prediction), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
exit()