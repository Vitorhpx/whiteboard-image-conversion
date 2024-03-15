import os
import cv2
import numpy as np

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)
from flask_cors import CORS, cross_origin
from image_recognition import convert_image, convert_image_v2, convert_image_v3, convert_image_v32, convert_image_v12 
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/img', methods=['GET','POST'])
@cross_origin()
def img_v1():
   image_file = request.files['image']
   image_data = image_file.read()
   image_array = np.frombuffer(image_data, dtype=np.uint8)
   return convert_image(image_array)


@app.route('/v12/img', methods=['GET','POST'])
@cross_origin()
def img_v12():
   image_file = request.files['image']
   image_data = image_file.read()
   image_array = np.frombuffer(image_data, dtype=np.uint8)
   return convert_image_v12(image_array)


@app.route('/v2/img', methods=['GET','POST'])
@cross_origin()
def img_v2():
   image_file = request.files['image']
   image_data = image_file.read()
   image_array = np.frombuffer(image_data, dtype=np.uint8)
   return convert_image_v2(image_array)

@app.route('/v3/img', methods=['GET','POST'])
@cross_origin()
def img_v3():
   image_file = request.files['image']
   image_data = image_file.read()
   image_array = np.frombuffer(image_data, dtype=np.uint8)
   return convert_image_v3(image_array)

@app.route('/v32/img', methods=['GET','POST'])
@cross_origin()
def img_v32():
   image_file = request.files['image']
   image_data = image_file.read()
   image_array = np.frombuffer(image_data, dtype=np.uint8)
   return convert_image_v32(image_array)
   
if __name__ == '__main__':
   app.run()
