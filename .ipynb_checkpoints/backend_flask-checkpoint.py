from deepface import DeepFace
import numpy as np
import os
import cv2
import speech_recognition as sr
import pyttsx3
import tensorflow as tf

from keras.models import model_from_json
import mediapipe as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from flask import Flask, render_template, Response, request,redirect,url_for
from gtts import gTTS
global graph
global writer
from skimage.transform import resize
facecascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

graph=tf.compat.v1.get_default_graph()
writer=None
model=load_model('model.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


vals=['A','B','C','D','E','F','G','H','I']
app=Flask(__name__,template_folder="template")
print("Accessing video stream")
app.static_folder = 'static'
vs=cv2.VideoCapture(0)
pred=""

def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
def generate_frames():
    while True:
        ## read the camera frame
        success, frame = vs.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(
                gray,
                1.1,4
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/sign_to_speech')
def sign_to_speech():
    return render_template('sign_to_speech.html')
@app.route('/speech_to_sign')
def speech_to_sign():
    return render_template('speech_to_sign.html')



@app.route('/video',methods=['GET', 'POST'])
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)