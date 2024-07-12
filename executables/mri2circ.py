from __future__ import generators

import logging
import glob, os, functools
import sys
sys.path.append('../')

import SimpleITK as sitk
from scipy.signal import medfilt
import numpy as np
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
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.zscore import ZScoreNormalize

import pandas as pd
import tensorflow as tf
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 
from compute_population_pred import compute_crop_line
  
from scripts.densenet_regression import DenseNet
from scripts.unet import get_unet_2D
from scripts.preprocess_utils import load_nii,save_nii, find_file_in_path, iou, enhance_noN4,crop_center, get_id_and_path
from scripts.feret import Calculater
from settings import  target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, scaling_factor
from scripts.infer_selection import get_slice_number_from_prediction, funcy
from scripts.preprocess_utils import closest_value,find_centile,find_exact_percentile_return_number,add_median_labels
import math
import warnings
import os
import traceback
import subprocess
from hmac import new

warnings.filterwarnings('ignore')

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) == 0:
    physical_devices = tf.config.experimental.list_physical_devices('CPU')
else:   
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def select_template_based_on_age(age):
    for golden_file_path, age_values in age_ranges.items():
        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
            print(golden_file_path)
            return golden_file_path

# compute the cropline
def compute_crop_line(img_input,infer_seg_array_2d_1,infer_seg_array_2d_2):
    binary = img_input>-1.7
    binary_smoothed = scipy.signal.medfilt(binary.astype(int), 51)
    img = binary_smoothed.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    img = cv2.drawContours(mask, contours, -1, (255),1)

    max_y,ind_max = 0,0
    min_y,ind_min = 512,0
    if len(contours)>0:
        for i in range(0,len(contours[0])):
            x,y = contours[0][i][0]
            if y<=min_y:
                min_y,ind_min = y,i
            if y>=max_y:
                max_y,ind_max = y,i
        crop_line = (contours[0][ind_min][0][0]+contours[0][ind_max][0][0])/2
        
        return crop_line
    else:
        return 100
    
# register the MRI to the template     
def register_to_template_cmd(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    if "nii" in input_image_path and "._" not in input_image_path:
        try:
            """
            return_code = subprocess.call("elastix -f "+fixed_image_path+" -m "+input_image_path+" -out "+\
            output_path + " -p ../shared_data/mni_templates/Parameters_Rigid.txt", shell=True,\
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            """
            return_code = subprocess.call("/Users/philipmattisson/Desktop/Centile/software/elastix-5.1.0-mac/bin/elastix -f " +
                                           fixed_image_path + " -m " + input_image_path + " -out " 
                                           + output_path + " -p /Users/philipmattisson/Desktop/Centile/software/git/itmt/shared_data/mni_templates/Parameters_Rigid.txt", 
                                           shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print(return_code)

            if return_code == 0:
                print("Registered ", rename_id)
                result_image = itk.imread(output_path+'/result.0.mhd',itk.F)
                itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
            else:
                print("Error registering ", rename_id)
                return_code = 1
        except Exception as e:
            print("is elastix installed?")
            print("An error occurred:", e)
            traceback.print_exc()
            return_code = 1
     
def filter_islands(muscle_seg):
    img = muscle_seg.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    cnt_mask = np.zeros(img.shape, np.uint8)
    area = 0
    c=0
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        mask = cv2.fillPoly(mask, pts=[c], color=(255, 0, 0))
        cnt_mask =  cv2.drawContours(cnt_mask, [c], -1, (255, 255, 255), 0)#cv.drawContours(cnt_mask, [c], 0, (255,255,0), 2)
    return mask, area, c


# change paths to your data here
#img_path = '../shared_data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz' # input raw t1 MRI path
#img_path = '/Users/philipmattisson/Desktop/Centile/data/TRUE_BRAIN/STD_HEAD_NECK_U-EXT_Head_82252_80448.nii'
img_path = '/Users/philipmattisson/Desktop/Centile/data/T1_UKB5563485.nii.gz'
age = 35 # age of subject
gender = "F" # gender
model_weight_path_segmentation = '../model/unet_models/test/itmt1.hdf5'
model_weight_path_selection = '../model/densenet_models/test/itmt1.hdf5'
path_to = "../output/" # save to

# MNI templates http://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/
age_ranges = {"../shared_data/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "../shared_data/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "../shared_data/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}
threshold = 0.75 # ie must be present on 3 out of 4 predictions


# Set the library path for the current script
os.environ['DYLD_LIBRARY_PATH'] = '/Users/philipmattisson/Desktop/Centile/software/elastix-5.1.0-mac/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

# load image
image, affine = load_nii(img_path)
plt.imshow(image[:,:,1])
print(nib.aff2axcodes(affine))

# path to store registered image in
new_path_to = path_to+img_path.split("/")[-1].split(".")[0]
if not os.path.exists(new_path_to):
    os.mkdir(new_path_to)

print(new_path_to)
#new_path_to = "/Users/philipmattisson/Desktop/Centile/software/git/itmt/output/test_output"
# register image to MNI template
golden_file_path = select_template_based_on_age(age)
#golden_file_path = "/Users/philipmattisson/Desktop/Centile/software/git/itmt/shared_data/mni_templates/nihpd_asym_04.5-08.5_t1w.nii"
print("Registering to template:", golden_file_path)

try:
    print(new_path_to)
    result = subprocess.run(["/Users/philipmattisson/Desktop/Centile/software/elastix-5.1.0-mac/bin/elastix", "-f", golden_file_path, "-m", img_path , "-out", new_path_to , "-p", "/Users/philipmattisson/Desktop/Centile/software/git/itmt/shared_data/mni_templates/Parameters_Rigid.txt"], capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode == 0:
        print("Registered ")
        result_image = itk.imread(new_path_to+'/result.0.mhd',itk.F)
        itk.imwrite(result_image, new_path_to+"/"+"registered"+".nii.gz")
    else:
        print("Error registering ", "with return code", result.returncode)
except Exception as e:
    print("An error occurred:", e)


#Load the registered image
# enchance and zscore normalize image
if not os.path.exists(new_path_to+"/no_z"):
    os.mkdir(new_path_to+"/no_z")
print(f"new_path_to {new_path_to}")
image_sitk =  sitk.ReadImage(new_path_to+"/registered.nii.gz")
#image_sitk = sitk.ReadImage("/Users/philipmattisson/Desktop/Centile/software/git/itmt/output/sub-pixar066_anat_sub-pixar066_T1w/registered.nii.gz")
image_array  = sitk.GetArrayFromImage(image_sitk)
image_array = enhance_noN4(image_array)
image3 = sitk.GetImageFromArray(image_array)

sitk.WriteImage(image3,new_path_to+"/no_z/registered_no_z.nii") 
cmd_line = "zscore-normalize "+new_path_to+"/no_z/registered_no_z.nii -o "+new_path_to+'/registered_z.nii'
subprocess.getoutput(cmd_line)     
print(cmd_line)
print("Preprocessing done!")


# load models
model_selection = DenseNet(img_dim=(256, 256, 1), 
                nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                compression_rate=0.5, sigmoid_output_activation=True, 
                activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
model_selection.load_weights(model_weight_path_selection)
print('\n','\n','\n','loaded:' ,model_weight_path_selection)  
    
model_unet = get_unet_2D(unet_classes,(target_size_unet[0], target_size_unet[1], 1),\
        num_convs=2,  activation='relu',
        compression_channels=[16, 32, 64, 128, 256, 512],
        decompression_channels=[256, 128, 64, 32, 16])
model_unet.load_weights(model_weight_path_segmentation)
print('\n','\n','\n','loaded:' ,model_weight_path_segmentation)


image_sitk = sitk.ReadImage(new_path_to+'/registered_z.nii')    
windowed_images  = sitk.GetArrayFromImage(image_sitk)

## predict slice
resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                            preserve_range=True, anti_aliasing=True, mode='constant')
series = np.dstack([resize_func(im) for im in windowed_images])
series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
series_n = []

for slice_idx in range(2, np.shape(series)[0]-2):
    im_array = np.zeros((256, 256, 1, 5))
    
    # create MIP of 5 slices = 5mm 
    im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
    im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
    im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
    im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
    im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
            
    im_array= np.max(im_array, axis=3)
            
    series_n.append(im_array)
    series_w = np.dstack([funcy(im) for im in series_n])
    series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])
        
predictions = model_selection.predict(series_w)
slice_label = get_slice_number_from_prediction(predictions)
print("Predicted slice:", slice_label)

## Inference and segmentation
img = nib.load(new_path_to+'/registered_z.nii')  
image_array, affine = img.get_fdata(), img.affine
infer_seg_array_3d_1,infer_seg_array_3d_2 = np.zeros(image_array.shape),np.zeros(image_array.shape)
print(np.asarray(nib.aff2axcodes(affine)))
threshold = 0.75

def euclidean(x, y):
    return math.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)

def get_contour(img_input):
    cnt, perimeter, max_cnt = 0, 0, 0
    binary = img_input > -1.7
    binary_smoothed = scipy.signal.medfilt(binary.astype(int), 51)
    img = binary_smoothed.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > cnt:
            cnt = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            max_cnt = contour  
            convexHull = cv2.convexHull(contour)
    
    p = 0
    for i in range(1, len(convexHull)):
        p += euclidean(convexHull[i][0], convexHull[i - 1][0])
    
    return round(perimeter, 2), round(p, 2)

image_array_2d = image_array[:, 15:-21, slice_label] 
perimeter_opencv, perimeter_convex = get_contour(image_array_2d)

print(f'perimeter_opencv {perimeter_opencv}, perimeter_convex {perimeter_convex}')










