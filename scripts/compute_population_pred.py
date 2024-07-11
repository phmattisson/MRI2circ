from __future__ import generators
import logging
import glob, os, functools
import sys
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
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
tf.get_logger().setLevel("ERROR")

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scripts.densenet_regression import DenseNet
from scripts.unet import get_unet_2D
from scripts.preprocess_utils import load_nii,save_nii, find_file_in_path,iou, enhance_noN4,crop_center, get_id_and_path
from scripts.feret import Calculater
from settings import target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, major_voting,scaling_factor
from scripts.infer_selection import get_slice_number_from_prediction, funcy
import warnings

## this is a file for computing the population prediction in batch

#path to ench and registered file
image_dir ='data/t1_mris/cbtn_reg_ench/z/' #'data/t1_mris/uscf_reg_ench/z/'
input_annotation_file = 'data/Dataset_cbtn.csv' #'data/Dataset_ucsf.csv'
output_dir = 'data/outputs/'
    
warnings.filterwarnings('ignore')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
# model paths                                   
model_weight_path_segmentation = 'model/unet_models/test/itmt1.hdf5'
model_weight_path_selection = 'model/densenet_models/test/itmt1.hdf5'

split_name = 'test'
measure_iou = True
threshold = 0.75 # ie must be present on 3 out of 4 predictions
alpha = 0.8 # for the plots blending value

list_true, list_pred = [], []
list_true_line,list_pred_line = [], []
list_csa = []

# get the metadata - age and sex
def get_metadata(row, image_dir):
    patient_id, image_path, age, gender = "","","",""
    patient_id = str(row['Filename']).split(".")[0]
    age =  row['AGE_M'] #// 12
    gender = row['SEX']
    dataset = row['dataset']
    path = find_file_in_path(patient_id, os.listdir(image_dir))
    
    patient_id =  patient_id.split("/")[-1]
    path = patient_id.split("/")[-1]    
    scan_folder = image_dir+path
    print(scan_folder)
    if os.path.exists(scan_folder):
        for file in os.listdir(scan_folder):
            t = image_dir+path+"/"+file
            if patient_id in file:
                image_path = t
    return patient_id, image_path, age, gender,dataset

# filter the islands in the segmentation mask
def filter_islands(muscle_seg):
    img = muscle_seg.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    #cnt_mask = np.zeros(img.shape, np.uint8)
    area = 0
    c=""
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        mask = cv2.fillPoly(mask, pts=[c], color=(255, 0, 0))
        #cnt_mask =  cv2.drawContours(cnt_mask, [c], -1, (255, 255, 255), 0)
    return mask.astype('bool'), area, c

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

## make 4 predictions over the same segment
def major_voting(image_array_2d, model_segmentation):
    img_half_11 = np.concatenate((image_array_2d[:,:256,:,:],np.zeros_like(image_array_2d[:,:256,:,:])),axis=1)
    img_half_21 = np.concatenate((np.zeros_like(image_array_2d[:,:256,:,:]),image_array_2d[:,:256,:,:]),axis=1)
    img_half_12 = np.concatenate((np.zeros_like(image_array_2d[:,256:,:,:]),image_array_2d[:,256:,:,:]),axis=1)
    img_half_22 = np.concatenate((image_array_2d[:,256:,:,:],np.zeros_like(image_array_2d[:,256:,:,:])),axis=1)

    flipped = np.flip(image_array_2d, axis=1)

    flipped_11 = np.concatenate((flipped[:,:256,:,:],np.zeros_like(flipped[:,:256,:,:])),axis=1)
    flipped_21 = np.concatenate((np.zeros_like(flipped[:,:256,:,:]),flipped[:,:256,:,:]),axis=1)
    flipped_12 = np.concatenate((np.zeros_like(flipped[:,256:,:,:]),flipped[:,256:,:,:]),axis=1)
    flipped_22 = np.concatenate((flipped[:,256:,:,:],np.zeros_like(flipped[:,256:,:,:])),axis=1)

    list_of_left_muscle = [img_half_11, img_half_21, flipped_12, flipped_22]
    list_of_right_muscle = [img_half_12,img_half_22, flipped_11, flipped_21]

    list_of_left_muscle_preds = []
    list_of_right_muscle_preds = []

    for image in list_of_left_muscle: 
        infer_seg_array = model_segmentation.predict(image)
        muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)               
        list_of_left_muscle_preds.append(muscle_seg)
                    
    for image in list_of_right_muscle: 
        infer_seg_array = model_segmentation.predict(image)
        muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)             
        list_of_right_muscle_preds.append(muscle_seg)
                
    list_of_left_muscle_preds_halved = [list_of_left_muscle_preds[0][:,:256,:,:],
                                    list_of_left_muscle_preds[1][:,256:,:,:],
                                    np.flip(list_of_left_muscle_preds[2][:,256:,:,:],axis=1),
                                    np.flip(list_of_left_muscle_preds[3][:,:256,:,:],axis=1)]

    list_of_right_muscle_preds_halved = [list_of_right_muscle_preds[0][:,256:,:,:],
                                    list_of_right_muscle_preds[1][:,:256,:,:],
                                    np.flip(list_of_right_muscle_preds[2][:,:256,:,:],axis=1),
                                    np.flip(list_of_right_muscle_preds[3][:,256:,:,:],axis=1)]
                
    left_half_result = np.mean(list_of_left_muscle_preds_halved, axis=0)<=threshold 
    right_half_result = np.mean(list_of_right_muscle_preds_halved, axis=0)<=threshold 
    muscle_seg_1 = np.concatenate((left_half_result,np.zeros_like(left_half_result)),axis=1)
    muscle_seg_2 = np.concatenate((np.zeros_like(left_half_result),right_half_result),axis=1)
    return muscle_seg_1, muscle_seg_2

## measuring TMT and CSA
def measure_tm(image_array,infer_seg_array_3d_1,infer_seg_array_3d_2,slice_label,crop_line):
    objL_pred_minf_line, objR_pred_minf_line, objL_pred_minf, objR_pred_minf = 0,0,0,0
    objL_pred_maxf_90, objR_pred_maxf_90 = 0,0
            
    if np.sum(infer_seg_array_3d_1[:100,:,slice_label])>2:
        objL_pred_minf = round(Calculater(infer_seg_array_3d_1[:100,:,slice_label], edge=True).minf,2)
        objL_pred_maxf_90 = round(Calculater(infer_seg_array_3d_1[:100,:,slice_label], edge=True).maxf90,2)

    if np.sum(infer_seg_array_3d_2[100:,:,slice_label])>2:
        objR_pred_minf = round(Calculater(infer_seg_array_3d_2[100:,:,slice_label], edge=True).minf,2)
        objR_pred_maxf_90 = round(Calculater(infer_seg_array_3d_2[100:,:,slice_label], edge=True).maxf90,2)
            
    CSA_PRED_TM1 = round(np.sum(infer_seg_array_3d_1[:100,:,slice_label]),2)
    CSA_PRED_TM2 = round(np.sum(infer_seg_array_3d_2[100:,:,slice_label]),2)
                    
    if np.sum(infer_seg_array_3d_1[:100,int(crop_line):,slice_label])>2:
        objL_pred_minf_line = round(Calculater(infer_seg_array_3d_1[:100,int(crop_line):,slice_label], edge=True).minf,2)

    if np.sum(infer_seg_array_3d_2[100:,int(crop_line):,slice_label])>2:
        objR_pred_minf_line = round(Calculater(infer_seg_array_3d_2[100:,int(crop_line):,slice_label], edge=True).minf,2)
                
    CSA_PRED_TM1_line = round(np.sum(infer_seg_array_3d_1[:100,int(crop_line):,slice_label]),2)
    CSA_PRED_TM2_line = round(np.sum(infer_seg_array_3d_2[100:,int(crop_line):,slice_label]),2)
            
    return [CSA_PRED_TM1,CSA_PRED_TM2,
            round(CSA_PRED_TM1+CSA_PRED_TM2,2), 
            round((CSA_PRED_TM1+CSA_PRED_TM2)/2,2), # "CSA PRED SUM" "CSA PRED AVG"
            objL_pred_minf, objR_pred_minf,
            round(objL_pred_minf + objR_pred_minf,2),
            round((objL_pred_minf + objR_pred_minf)/2,2), #  "TMT PRED SUM"  "TMT PRED AVG"
            
            objL_pred_maxf_90, objR_pred_maxf_90,
            round(objL_pred_maxf_90 + objR_pred_maxf_90,2),
            round((objL_pred_maxf_90 + objR_pred_maxf_90)/2,2), #  "TMT PRED SUM 90"  "TMT PRED AVG 90"
            
            CSA_PRED_TM1_line, CSA_PRED_TM2_line,
            round(CSA_PRED_TM1_line + CSA_PRED_TM2_line,2),
            round((CSA_PRED_TM1_line + CSA_PRED_TM2_line)/2,2), #"CSA PRED SUM w line", #"CSA PRED AVG w line",
            objL_pred_minf_line, objR_pred_minf_line,
            round(objL_pred_minf_line + objR_pred_minf_line,2),
            round((objL_pred_minf_line + objR_pred_minf_line)/2,2)] # "TMT PRED SUM w line", # "TMT PRED AVG w line",
                            
 
if __name__=="__main__":
    # load models
    model_selection = DenseNet(img_dim=(256, 256, 1), 
                    nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                    compression_rate=0.5, sigmoid_output_activation=True, 
                    activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True)
    model_selection.load_weights(model_weight_path_selection)
    print('\n','\n','\n','loaded:' ,model_weight_path_selection)  
        
    model_segmentation = get_unet_2D(unet_classes,(target_size_unet[0], target_size_unet[1], 1),\
            num_convs=2,  activation='relu',
            compression_channels=[16, 32, 64, 128, 256, 512],
            decompression_channels=[256, 128, 64, 32, 16])
    model_segmentation.load_weights(model_weight_path_segmentation)
    print('\n','\n','\n','loaded:' ,model_weight_path_segmentation)  
    
    # load metadata file     
    df = pd.read_csv(input_annotation_file, header=0)
    print("Len of dataset:",df.shape[0])

    for idx in range(0,df.shape[0]):
        row = df.iloc[idx]
        patient_id, image_path, age, gender, dataset = get_metadata(row, image_dir)
        print(idx,patient_id, image_path, age, gender)
        
        if len(image_path)>3 and age > 3:
            image_sitk = sitk.ReadImage(image_path)    
            windowed_images  = sitk.GetArrayFromImage(image_sitk)   
        
            # enchancing is done in the preprocessing step
            resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                                preserve_range=True, anti_aliasing=True, mode='constant')
            series = np.dstack([resize_func(im) for im in windowed_images])
            series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
            series_n = []

            for slice_idx in range(2,np.shape(series)[0]-2):
                im_array = np.zeros((256,256,1,5))
                    
                im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
                im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
                im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
                im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
                im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
                    
                im_array = np.max(im_array, axis=3)
                    
                series_n.append(im_array)
                series_w = np.dstack([funcy(im) for im in series_n])
                series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])

            predictions = model_selection.predict(series_w)
            slice_label = get_slice_number_from_prediction(predictions)
            
            ## Unet
            img = nib.load(image_path)  
            image_array, affine = img.get_fdata(), img.affine
            infer_seg_array_3d_1,infer_seg_array_3d_2 = np.zeros(image_array.shape),np.zeros(image_array.shape)
            infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered = np.zeros(image_array.shape),np.zeros(image_array.shape)
            infer_seg_array_3d_merged_filtered =  np.zeros(image_array.shape)
            
            # rescale image into 512x512 for unet 
            image_array_2d = rescale(image_array[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
                
            print(major_voting)
            if major_voting == False:
                img_half_1 = np.concatenate((image_array_2d[:,:256,:,:],np.zeros_like(image_array_2d[:,:256,:,:])),axis=1)
                img_half_2 = np.concatenate((np.zeros_like(image_array_2d[:,256:,:,:]),image_array_2d[:,256:,:,:]),axis=1)

                infer_seg_array_1 = model_segmentation.predict(img_half_1)
                infer_seg_array_2 = model_segmentation.predict(img_half_2)

                muscle_seg_1 = (infer_seg_array_1[:,:,:,1] <= softmax_threshold).reshape(1,target_size_unet[0],target_size_unet[1],1)            
                muscle_seg_2 = (infer_seg_array_2[:,:,:,1] <= softmax_threshold).reshape(1,target_size_unet[0],target_size_unet[1],1)          
            else:
                muscle_seg_1, muscle_seg_2 = major_voting(image_array_2d, model_segmentation)
                
            
            # filter islands
            muscle_seg_1_filtered, area_1, cnt_1 = filter_islands(muscle_seg_1[0])
            muscle_seg_2_filtered, area_2, cnt_2 = filter_islands(muscle_seg_2[0])

            ## save plots 
            if os.path.exists(output_dir+"/pics/")==False:
                os.mkdir(output_dir+"/pics/")
            if os.path.exists(output_dir+"/pics/no_masks/")==False:
                os.mkdir(output_dir+"/pics/no_masks/")
            if os.path.exists(output_dir+"/pics/masks/")==False:
                os.mkdir(output_dir+"/pics/masks/")
            if os.path.exists(output_dir+"/pics/niftis/")==False:
                os.mkdir(output_dir+"/pics/niftis/")
                
            fg = plt.figure(figsize=(5, 5), facecolor='k')
            I = cv2.normalize(image_array_2d[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(output_dir+"/pics/no_masks/"+str(dataset)+"_"+patient_id+".png", I)
            im = cv2.imread(output_dir+"/pics/no_masks/"+str(dataset)+"_"+patient_id+".png")                        
            im_copy = im.copy()
            
            result = im.copy()
            for cont in [cnt_1,cnt_2]: 
                if len(cont)!=0:
                    if cv2.contourArea(cont) <= 1:
                        im_copy = cv2.drawContours(im_copy, [cont], -1, (0, 0, 255), -1)
                    else:
                        im_copy = cv2.drawContours(im_copy, [cont], -1, (51, 197, 255), -1)
            filled = cv2.addWeighted(im, alpha, im_copy, 1-alpha, 0)
            for cont in [cnt_1,cnt_2]: 
                if len(cont)!=0:
                    if cv2.contourArea(cont) <= 1:
                        result = cv2.drawContours(filled, [cont], -1, (0, 0, 255), 0)
                    else:
                        result = cv2.drawContours(filled, [cont], -1, (51, 197, 255), 0)

            cv2.imwrite(output_dir+"/pics/masks/"+str(dataset)+"_"+patient_id+"_mask.png", result)
            
            # rescale for the unet
            infer_seg_array_2d_1 = rescale(muscle_seg_1[0],1/scaling_factor)
            infer_seg_array_2d_2 = rescale(muscle_seg_2[0],1/scaling_factor)
            infer_seg_array_2d_1_filtered = rescale(muscle_seg_1_filtered,1/scaling_factor)
            infer_seg_array_2d_2_filtered = rescale(muscle_seg_2_filtered,1/scaling_factor)

            # save to 3d
            infer_seg_array_3d_1[:,:,slice_label] = np.pad(infer_seg_array_2d_1[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
            infer_seg_array_3d_2[:,:,slice_label] = np.pad(infer_seg_array_2d_2[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
            infer_seg_array_3d_1_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_1_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
            infer_seg_array_3d_2_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_2_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
            
            concated = np.concatenate((infer_seg_array_2d_1_filtered[:100,:,0],infer_seg_array_2d_2_filtered[100:,:,0]),axis=0)    
            infer_seg_array_3d_merged_filtered[:,:,slice_label] = np.pad(concated,[[0,0],[15,21]],'constant',constant_values=0)
            infer_3d_path = output_dir+"/pics/niftis/"+str(dataset)+"_"+patient_id + '_AI_seg.nii.gz'
            save_nii(infer_seg_array_3d_merged_filtered, infer_3d_path, affine)
                
            # measure TM
            if measure_iou:
                crop_line = compute_crop_line(image_array[:,15:-21,slice_label],infer_seg_array_2d_1,infer_seg_array_2d_2)
                unfiltered_metrics = measure_tm(image_array,infer_seg_array_3d_1,infer_seg_array_3d_2,slice_label,crop_line)
                filtered_metrics = measure_tm(image_array,infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered,slice_label,crop_line)
                list_csa.append([patient_id, gender, age, dataset, slice_label, *unfiltered_metrics, *filtered_metrics])
                
            gc.collect()
           
    # save metrics 
    print(list_csa)
    df = pd.DataFrame(np.asarray(list_csa))
    df.to_csv(output_dir+"csa_population_"+str(dataset)+".csv", 
                                            header=["ID",
                                                    "Gender", 
                                                    "Age",
                                                    "Dataset",
                                                    "Slice label",
                                                    
                                                ##unfiltered
                                                "CSA PRED TM1","CSA PRED TM2",
                                                "CSA PRED SUM", 
                                                "CSA PRED AVG",
                                                "TM PRED1","TM PRED2",
                                                "TMT PRED SUM", 
                                                "TMT PRED AVG", 
                                                
                                                "TM PRED1 90","TM PRED2 90",
                                                "TMT PRED SUM 90", 
                                                "TMT PRED AVG 90", 
                                                
                                                "CSA PRED TM1 w line", 
                                                "CSA PRED TM2 w line",
                                                "CSA PRED SUM w line",
                                                "CSA PRED AVG w line",
                                                "TM PRED1 w line",
                                                "TM PRED2 w line",
                                                "TMT PRED SUM w line",
                                                "TMT PRED AVG w line",
                                                
                                                ## filtered
                                                "CSA PRED TM1 filtered","CSA PRED TM2 filtered",
                                                "CSA PRED SUM filtered", 
                                                "CSA PRED AVG filtered",
                                                "TM PRED1 filtered", "TM PRED2 filtered",
                                                "TMT PRED SUM filtered", 
                                                "TMT PRED AVG filtered", 
                                                
                                                "TM PRED1 90 filtered ", "TM PRED2 90 filtered",
                                                "TMT PRED SUM 90 filtered", 
                                                "TMT PRED AVG 90 filtered", 
                                                
                                                "CSA PRED TM1 w line filtered", 
                                                "CSA PRED TM2 w line filtered",
                                                "CSA PRED SUM w line filtered",
                                                "CSA PRED AVG w line filtered",
                                                "TM PRED1 w line filtered",
                                                "TM PRED2 w line filtered",
                                                "TMT PRED SUM w line filtered",
                                                "TMT PRED AVG w line filtered"
                                                ])
        