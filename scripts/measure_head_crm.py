from __future__ import generators
import logging
import glob, os, functools
import sys

import SimpleITK as sitk
from scipy.signal import medfilt
import numpy as np
import math
from numpy import median
import scipy
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
from skimage.transform import resize,rescale
import cv2
import itk
import subprocess

import pandas as pd
import tensorflow as tf
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scripts.densenet_regression import DenseNet
from scripts.unet import get_unet_2D
from scripts.preprocess_utils import load_nii,enhance, save_nii, find_file_in_path,iou, enhance_noN4,crop_center, get_id_and_path
from scripts.feret import Calculater
from settings import  target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, major_voting,scaling_factor
from scripts.infer_selection import get_slice_number_from_prediction, funcy
import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
# calculate the euclidean distance between two points
def euclidean(x,y):
    return math.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)

# paths to the datasets
dict_paths = {
    'ABCD': "data/t1_mris/registered",
    'ABIDE': "data/t1_mris/abide_ench_reg",
    'AOMIC':"data/t1_mris/aomic_reg_ench",
    'BABY': "data/t1_mris/registered",
    'Calgary': "data/t1_mris/calgary_reg_ench",
    'ICBM': "data/t1_mris/icbm_ench_reg",
    'IXI':"data/t1_mris/registered",
    'NIMH': "data/t1_mris/nihm_ench_reg",
    'HIMH': "data/t1_mris/nihm_ench_reg",
    'PING': "data/t1_mris/pings_ench_reg",
    'Pixar': 'data/t1_mris/pixar_ench',
    'SALD': "data/t1_mris/sald_reg_ench",
    'NYU': "data/t1_mris/nyu_reg_ench",
    'Healthy adults': "data/t1_mris/healthy_adults_nihm_reg_ench",
    'HAN': "data/t1_mris/healthy_adults_nihm_reg_ench",
    'Petfrog':"data/t1_mris/petfrog_reg_ench"}

# get the path to the image
def find_path(img_id, dataset):
    dataset_image_path = dict_paths[dataset]
    for image_path in os.listdir(dataset_image_path+"/no_z"):
        if img_id in image_path:
            image_path_no_z = dataset_image_path+"/no_z/"+image_path
            image_path_z = dataset_image_path+"/z/"+img_id+"/"+img_id+".nii"
    patient_id = str(img_id).split(".")[0]  
    return image_path_z, image_path_no_z, patient_id

# compute the crm
def get_contour(img_input): 
    cnt,perimeter,max_cnt = 0,0,0
    # normalize the image
    binary = img_input>-1.7
    binary_smoothed = scipy.signal.medfilt(binary.astype(int), 51)
    img = binary_smoothed.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    
    # get the contour with the largest area
    for contour in contours:
        if cv2.contourArea(contour)>cnt:
            cnt = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour,True)
            max_cnt = contour  
            convexHull = cv2.convexHull(contour)
    
    # compute the perimeter of the convex hull
    p = 0
    for i in range(1,len(convexHull)):
        p = p + euclidean(convexHull[i][0],convexHull[i-1][0])
    print("Perim of the convex hull",p)
    
    return round(perimeter,2), round(p,2)

input_annotation_file = 'data/pop_norms.csv'
df = pd.read_csv(input_annotation_file, header=0,)

# adapt mri source path 
lst_crm = []
lst_errors = []
print(df.shape[0])
for i in range(0, df.shape[0]):
    try:
        mri_source_path = dict_paths[df['Dataset'].iloc[i]]+"/z/"
        
        for image_path in os.listdir(mri_source_path):
            if df['ID'].iloc[i] in image_path:   
                image_path = mri_source_path+df['ID'].iloc[i]+"/"+os.listdir(mri_source_path+df['ID'].iloc[i])[0]
                print(image_path)
                
                slice_label = df['Slice label'].iloc[i]
                print("slice_label",slice_label)
                
                img = nib.load(image_path)  
                image_array, affine = img.get_fdata(), img.affine
                image_array_2d = image_array[:,15:-21,slice_label] 
                perimeter_opencv,perimeter_convex = get_contour(image_array_2d)
                
                lst_crm.append([df['ID'].iloc[i],df['Age'].iloc[i],df['Gender'].iloc[i],df['Dataset'].iloc[i],
                                perimeter_opencv,perimeter_convex, 
                                df['TMT PRED AVG filtered'].iloc[i], df['CSA PRED AVG filtered'].iloc[i],
                                df['TMT PRED AVG'].iloc[i], df['CSA PRED AVG'].iloc[i],slice_label])
                break
    except:
        print("error occured on image",df['ID'].iloc[i])
        lst_errors.append(df['ID'].iloc[i])
        continue
        
df = pd.DataFrame(lst_crm)
df.to_csv(path_or_buf= "data/dataset_measured_heads_filtered.csv", header=['ID','Age','Gender','Dataset',
                                                                  "perimeter_opencv","perimeter_convex",
                                                                    "TMT PRED AVG filtered","CSA PRED AVG filtered",
                                                                    "TMT PRED AVG","CSA PRED AVG","slice_label"])
df = pd.DataFrame(lst_errors)
df.to_csv(path_or_buf= "data/dataset_measured_heads_errors_filtered.csv")

