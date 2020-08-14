# -*- coding: utf-8 -*-
"""
@author: Jaskirat Singh
"""

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import sys
import uuid
import urllib.request
from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from data_utils import auto_body_crop

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__, static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/news.html')
def news():
    return render_template('news.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/faqs.html')
def faqs():
    return render_template('faqs.html')


@app.route('/prevention.html')
def prevention():
    return render_template('prevention.html')


@app.route('/upload.html')
def upload():
    return render_template('upload.html')


@app.route('/upload_chest.html')
def upload_chest():
    return render_template('upload_chest.html')


@app.route('/upload_ct.html')
def upload_ct():
    return render_template('upload_ct.html')


frozen_graph = "models/saved_model.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
class GradCAM:
    def __init__(self, graph, classes, outLayer, targetLayer=None):
        self.graph = graph
        self.classes = classes
        self.targetLayer = targetLayer
        self.outLayer = outLayer

        if self.targetLayer is None:
            self.target = self.find_target_tensor()
        else:
            self.target = self.graph.get_tensor_by_name(self.targetLayer)

    def find_target_tensor(self):
        """
        Find the last tensor that have 4D shape if targetLayer is not specified.
        :return:
        """
        tensor_names = [t.name for op in tf.get_default_graph().get_operations() for t in op.values() if
                   "save" not in str(t.name)]
        for tensor_name in reversed(tensor_names):
            tensor = self.graph.get_tensor_by_name(tensor_name)
            if len(tensor.shape) == 4:
                return tensor

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_grads(self):
        results = {} # grads of classes with keys being classes and values being normalized gradients
        for classIdx in self.classes:
            one_hot = tf.sparse_to_dense(classIdx, [len(self.classes)], 1.0)
            signal = tf.multiply(self.graph.get_tensor_by_name(self.outLayer),one_hot)
            loss = tf.reduce_mean(signal)

            grads = tf.gradients(loss, self.target)[0]

            norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads)))+tf.constant(1e-5))

            results[classIdx] = norm_grads

        return results
    
def generate_cam(conv_layer_out, grads_val, upsample_size):
    weights = np.mean(grads_val, axis=(0,1))
    cam = np.zeros(conv_layer_out.shape[0:2], dtype=np.float32)

    # Weight averaginng
    for i, w in enumerate(weights):
        cam += w*conv_layer_out[:,:,i]

    # Apply reLU
    cam = np.maximum(cam, 0)
    cam = cam/np.max(cam)
    cam = cv2.resize(cam, upsample_size)

    # Convert to 3D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3,[1,1,3])

    return cam3

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}



@app.route('/uploaded_chest', methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('upload.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect('upload.html')
        if file:
            filename = secure_filename(file.filename)
            file_path =os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)

            origin_im = cv2.imread(file_path)  # read file
            origin_im = cv2.cvtColor(origin_im, cv2.COLOR_BGR2RGB)
            x=origin_im
            h, w, c = x.shape
            offset = int(x.shape[0] * 0.08)
            x=x[offset:]
            size = min(x.shape[0], x.shape[1])
            offset_h = int((x.shape[0] - size) / 2)
            offset_w = int((x.shape[1] - size) / 2)
            x=x[offset_h:offset_h + size, offset_w:offset_w + size]
            x = cv2.resize(x,(480,480))
            x = x.astype('float32') / 255.0
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def, 
                    input_map=None, 
                    return_elements=None, 
                    name="", 
                    #op_dict=None, 
                    #producer_op_list=None
                    )
                image_tensor = graph.get_tensor_by_name("input_1:0")
                pred_tensor  = graph.get_tensor_by_name("norm_dense_1/Softmax:0")
                sess= tf.Session(graph=graph)
                gradCam = GradCAM(graph=graph, classes = [0,1,2], outLayer="norm_dense_1/Softmax:0", targetLayer="conv5_block3_out/add:0")
                grads = gradCam.compute_grads()
                size_upsample = (w,h)

                pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
                
                output, grads_val = sess.run([gradCam.target, grads[pred.argmax(axis=1)[0]]], feed_dict={image_tensor: np.expand_dims(x, axis=0)})
                cam3 = generate_cam(output[0],grads_val[0],size_upsample)
                
                # Overlay cam on image
                cam3 = np.uint8(255*cam3)
                cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
                new_im = cam3*0.3 + origin_im*0.5
                filename1 = my_random_string(10) + filename
                filename2 = my_random_string(12) + filename
                os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),new_im)
                print("GradCAM image saved ")

            print(pred)
            inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
            pred_class = inv_mapping[pred.argmax(axis=1)[0]]
            pred_proba = "{:.2f}".format((pred.max(axis=1)[0])*100)
            print(pred_proba)
            print(pred_class)
            result = pred_class.capitalize()
            # check if models exists
            return render_template('results_chest.html', result=result, probability=pred_proba,imagesource='/assets/images/'+filename1,imagesource1='/assets/images/'+filename2)

frozen_graph1 = "models/saved_model_CT.pb"
with tf.gfile.GFile(frozen_graph1, "rb") as f:
    graph_def1 = tf.GraphDef()
    graph_def1.ParseFromString(f.read())
    
def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.
    

@app.route('/uploaded_ct', methods=['POST', 'GET'])
def uploaded_ct():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('upload.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect('upload.html')
        if file:
            filename = secure_filename(file.filename)
            file_path =os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
             # Load and preprocess image
            origin_im = cv2.imread(file_path)  # read file
            origin_im = cv2.cvtColor(origin_im, cv2.COLOR_BGR2RGB)
            print(origin_im.shape)
            h, w, c = origin_im.shape
            image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)  # read file
            image, _ = auto_body_crop(image)
            image = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(np.stack((image, image, image), axis=-1), axis=0)
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def1, 
                    input_map=None, 
                    return_elements=None, 
                    name="", 
                    #op_dict=None, 
                    #producer_op_list=None
                    )
                image_tensor = graph.get_tensor_by_name("Placeholder:0")
                pred_tensor  = graph.get_tensor_by_name("softmax_tensor:0")
                TRAINING_PH_TENSOR= graph.get_tensor_by_name("is_training:0")

                sess= tf.Session(graph=graph)
                gradCam = GradCAM(graph=graph, classes = [0,1,2], outLayer="softmax_tensor:0", targetLayer="resnet_model/add_15:0")
                grads = gradCam.compute_grads()
                size_upsample = (w,h)
                 # Create feed dict
                feed_dict = {image_tensor: image, TRAINING_PH_TENSOR: False}
                pred = sess.run(pred_tensor, feed_dict)
                
                output, grads_val = sess.run([gradCam.target, grads[pred.argmax(axis=1)[0]]], feed_dict)
                cam3 = generate_cam(output[0],grads_val[0],size_upsample)
                
                # Overlay cam on image
                cam3 = np.uint8(255*cam3)
                cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
                new_im = cam3*0.3 + origin_im*0.5
                filename1 = my_random_string(10) + filename
                filename2 = my_random_string(12) + filename
                os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),new_im)
                print("GradCAM CT image saved ")
                
            print(pred)
            inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
            pred_class = inv_mapping[pred.argmax(axis=1)[0]]
            pred_proba = "{:.2f}".format((pred.max(axis=1)[0])*100)
            print(pred_proba)
            print(pred_class)
            result = pred_class.capitalize()    
            return render_template('results_ct.html',result=result, probability=pred_proba,imagesource='/assets/images/'+filename1,imagesource1='/assets/images/'+filename2)
        
if __name__ == '__main__':
    app.secret_key = ".."
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True,host='127.0.0.1', port=port)
