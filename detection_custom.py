#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-08-14
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, detect_image_and_save_count, detect_video, detect_realtime
from yolov3.configs import *

image_path   = "./IMAGES/e0e62403f.jpg"
video_path   = "./IMAGES/test.mp4"

if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

#yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
#load_yolo_weights(yolo, Darknet_weights) # use Darknet weights

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights
#image = detect_image(yolo, image_path, "one.jpg", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

'''
if YOLO_FRAMEWORK == "tf": # TensorFlow detection
    if YOLO_TYPE == "yolov4":
        Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
    if YOLO_TYPE == "yolov3":
        Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
    yolo.load_weights(YOLO_CUSTOM_WEIGHTS) # use custom weights
    
elif YOLO_FRAMEWORK == "trt": # TensorRT detection
    saved_model_loaded = tf.saved_model.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    yolo = saved_model_loaded.signatures['serving_default']

'''
import pandas as pd
from tqdm import tqdm
# os.mkdir('./Results_0.45/')
# for i in tqdm(os.listdir('./custom_dataset/test/')):
#     detect_image(yolo, './custom_dataset/test/'+i, "./Results_0.45/"+i, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

# os.mkdir('./Results_count/')
df = pd.DataFrame(columns = ['Name', 'truth', 'pred'])
truth = list(open('./model_data/wheat_test.txt', 'r'))

i = 0
while(True):
    try:
        image = truth[i].split()
        image_truth_len = len(image)-1
        image_name = image[0].split('/')[-1]
        pred_len = detect_image_and_save_count(yolo, './custom_dataset/test/'+image_name, "./Results_count/"+image_name, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        df.loc[i] = {'Name':image_name, 'truth': image_truth_len, 'pred': pred_len}
        i+=1
        print(i)
    except:
        break
df.to_csv('count_truth_vs_pred.csv')


#detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

#detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

