# Copyright 2017 AutoNxt Automation Pvt. Ltd. and Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' Attendance System based on Transfer Learning on Google's Inception V3 labels
    Train your labels based on the instructions given here:
    https://www.tensorflow.org/tutorials/image_retraining
    
    This script loads the graph 'output_graph.pb' and feeds in the camera stream
    The region of interest i.e the user's face is extracted from the frame and
    converted to a numpy array. This array is passed to the softmax_tensor as
    input which calculates the predictions from the final layer softmax function.
'''

import cv2 as opencv
import os
import numpy as np
import tensorflow as tf


def get_labels():
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'labels/output_labels.txt')), 'r') as file:
        labels = [line.rstrip('\n') for line in file]
    return labels


labels = get_labels()


def run_detection(image_array, labels):
    with tf.gfile.FastGFile(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classifiers/output_graph.pb')),
                            'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())
        _ = tf.import_graph_def(graph_def=graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': image_array})
        prediction = predictions[0]

        prediction = prediction.tolist()
        max_value = max(prediction)
        predicted_person = labels[prediction.index(max_value)]

        print("%s (%.2f%%)" % (predicted_person, max_value * 100))


face_cascade = opencv.CascadeClassifier(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classifiers/haarcascade_frontalface_default.xml')))
video_cam = opencv.VideoCapture(0)
while True:
    status, frame = video_cam.read()
    body = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)
    for (x, y, width, height) in body:
        region_of_interest = frame[y:y + height, x:x + width]
        if height > 100 or width > 100:
            run_detection(np.asarray(region_of_interest), labels)
