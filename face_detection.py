# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import cv2
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# silence warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CHANNEL = 3
image_1_path = '/content/drive/MyDrive/NB/input/input00.txt'


def txt_to_image(image_path):
  """takes in a path to the txt file containing the image pixels and returns an image in array form"""
  with open(image_path, "r") as file:  # read the data from the text file
    lines = file.readlines()
    image_list = [[line.replace(" ", ",").split(",")] for line in lines[1:]]
  first_line = lines[0]
  first_line = first_line.split()
  R = int(first_line[0]) # rows
  C = int(first_line[1]) # columns
  image = np.array(image_list)
  image = image.reshape(R, C, CHANNEL).astype(np.uint8) 
  image = image[:,:,::-1] # transform to RGB
  return image



def haarcascades_facedetector(image):
  """takes in an image and applies haarcascades face detector 
  and returns number of faces detected"""
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert image to grayscale
  faceCascade = cv2.CascadeClassifier('/content/drive/MyDrive/NB/haarcascade_frontalface_default.xml')
  haar_faces = faceCascade.detectMultiScale(gray, 
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(20, 20))
  faces = len(haar_faces)
  return faces



def mtcnn_facedetector(image):
  """takes in an image and applies mtcnn face detector 
  and returns number of faces detected"""
  # create the detector, using default weights
  detector = MTCNN()
  mtcnn_faces = detector.detect_faces(image)
  faces = len(mtcnn_faces)
  return faces
