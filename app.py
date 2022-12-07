import io
from operator import truediv
import os
import json
from PIL import Image
import subprocess
from werkzeug.utils import secure_filename, send_from_directory
from webcam import Webcam

import cv2
import numpy as np

import torch
from flask import Flask, flash,jsonify, url_for, render_template, request, redirect, send_file, Response


app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER


webcam = Webcam()

# finds the model inside your directory - works only if there is one model

def find_model():
    for f  in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")
    
model_name = find_model()
model =torch.hub.load("WongKinYiu/yolov7", 'custom',model_name)

model.eval()

# function get prediction from input imgages
def get_prediction(img_bytes):
    output = io.BytesIO(img_bytes)
    output.seek(0)
    img = Image.open(output)
    imgs = [img] #batched list of images
    # Inference
    results =model(imgs, size=640)# includes NMS
    output.flush()
    return results

# inference image
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
          
        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir='static')
        filename = secure_filename('image0.jpg')
        
        return render_template('result.html',result_image = filename,model_name = model_name)

    return render_template('index.html')

# stream from webcam
@app.route('/webcam')
def webcam_index():
    return render_template("webcam.html")

def read_from_webcam():
        while True:
            #read from webcam
            img_bytes = next(webcam.get_frame())
            
            #get prediction
            predict = get_prediction(img_bytes)
            predict.save(os.path.join(app.config['RESULT_FOLDER']))
            results = open(app.config['RESULT_FOLDER'] + '/image0.jpg','rb').read()
     
            #return to web
            yield b'Content-Type: image/jpeg\r\n\r\n' + results + b'\r\n--frame\r\n'

@app.route('/image_feed')
def image_feed():
   return Response(read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')