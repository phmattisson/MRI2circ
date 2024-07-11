import sys
import os
import gc
sys.path.append('../TM2_segmentation')

import SimpleITK as sitk
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.transform import resize, rescale
from scipy.ndimage import rotate
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy import median
import skimage
import cv2

from scripts.preprocess_utils import load_nii, save_nii, iou, crop_center, find_file_in_path, get_id_and_path
from scripts.unet import get_unet_2D
from scripts.feret import Calculater

# compute the ICC for the segmentation results
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

# compute the mean dice coefficient
def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width)
    batch_size = y_true.shape[0]
    mean_dice_channel = 0.
    for i in range(batch_size):
        channel_dice = single_dice_coef(y_true[i, :, :], y_pred_bin[i, :, :])
        mean_dice_channel += channel_dice/(batch_size)
    return mean_dice_channel

# compute the median dice coefficient
def median_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width)
    batch_size = y_true.shape[0]
    median_dice_channel = []
    for i in range(batch_size):
        channel_dice = single_dice_coef(y_true[i, :, :], y_pred_bin[i, :, :])
        median_dice_channel.append(channel_dice)
    t=sorted(median_dice_channel)   
    return median(t)

# compute the ICC for the segmentation results
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

# helper function to get the age and sex
def get_metadata(row, image_dir):
    patient_id, image_path, age, gender = "","","",""
    patient_id = str(row['Filename']).split(".")[0]
    age =  row['AGE_M'] // 12
    gender = row['Sex']
    path = find_file_in_path(patient_id, os.listdir(image_dir))
    
    patient_id =  patient_id.split("/")[-1]
    path = patient_id.split("/")[-1]    
    scan_folder = image_dir+path

    if os.path.exists(scan_folder):
        for file in os.listdir(scan_folder):
            t = image_dir+path+"/"+file
            if patient_id in file:
                image_path = t
    return age, gender

# compute the IOU
def iou(component1, component2):
    component1 = np.array(component1, dtype=bool)
    component2 = np.array(component2, dtype=bool)

    overlap = component1 * component2 # Logical AND
    union = component1 + component2 # Logical OR

    IOU = overlap.sum()/float(union.sum())
    return IOU

# set paths
annotator_1_healthy_path = "data/curated_test/final_test/z/"
annotator_1_cancer_path = 'data/mni_templates_BK/'
annotator_2_healthy_path = "data/curated_test/TM_validation_Kevin/healthy/"
annotator_2_cancer_path = "data/curated_test/TM_validation_Kevin/cancer/"

metadata_1_path = 'data/curated_test/final_test/Dataset_test_rescaled.csv'
metadata_2_path = "data/icc_val/cancer_meta.csv"
output_dir = "data/icc_val/"

# load the mask
def load_mask(tm_file):
    seg_data, seg_affine = load_nii(tm_file)
    if (np.asarray(nib.aff2axcodes(seg_affine))==['R', 'A', 'S']).all():
        slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
    elif (np.asarray(nib.aff2axcodes(seg_affine))==['L', 'P', 'S']).all():
        ornt = np.array([[0, -1],
                                [1, 1],
                                [2, 1]])
                
        seg = nib.load(tm_file)            
        seg = seg.as_reoriented(ornt)        
        seg_data, seg_affine  = seg.get_fdata(), seg.affine
                
        slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
            
    return seg_data[:,:,slice_label],slice_label, seg_data
    
    
def icc_masks(slice_csv_path,annotator_1_path, annotator_2_path, output_dir):
    df_prediction = pd.read_csv(slice_csv_path,index_col=0)

    split_name = 'test'
    alpha = 0.8 
    df_prediction=df_prediction[df_prediction['Ok registered? Y/N']=='Y'].reset_index()
    results_list = []
    
    print("Testing n=",len(df_prediction))
    for idx, row in df_prediction.iterrows():
        patient_id, image_path1, tm_file1,_ = get_id_and_path(row, annotator_1_path, True)
        patient_id, image_path2, tm_file2,_ = get_id_and_path(row, annotator_2_path, True)
        print(tm_file1, tm_file2)
        
        if len(image_path1) != 0:
            seg_mask1_2d, label1, seg_mask1_3d = load_mask(tm_file1)
            
        if len(image_path2) != 0:
            seg_mask2_2d, label2, seg_mask2_3d  = load_mask(tm_file2)
        
        if len(image_path1) != 0 and len(image_path2) != 0:
            slice_diff = abs(label1 - label2)
            dice_diff1 = round(dice_coef(seg_mask1_2d[100:,:], seg_mask2_2d[100:,:]),2)
            dice_diff2 = round(dice_coef(seg_mask1_2d[:100,:], seg_mask2_2d[:100,:]),2)
            dice_diff = round(dice_coef(seg_mask1_2d, seg_mask2_2d),2)
            iou_diff = round(iou(seg_mask1_2d, seg_mask2_2d),2)
            
            #measure TMT diff
            objR_pred_minf_l1 = round(Calculater(seg_mask1_3d[100:,:,label1], edge=True).minf,2)
            objR_pred_maxf_90_l1 = round(Calculater(seg_mask1_3d[100:,:,label1], edge=True).maxf90,2)
            objR_pred_minf_r1 = round(Calculater(seg_mask1_3d[:100,:,label1], edge=True).minf,2)
            objR_pred_maxf_90_r1 = round(Calculater(seg_mask1_3d[:100,:,label1], edge=True).maxf90,2)
            
            objR_pred_minf_l2 = round(Calculater(seg_mask2_3d[100:,:,label2], edge=True).minf,2)
            objR_pred_maxf_90_l2 = round(Calculater(seg_mask2_3d[100:,:,label2], edge=True).maxf90,2)
            objR_pred_minf_r2 = round(Calculater(seg_mask2_3d[100:,:,label2], edge=True).minf,2)
            objR_pred_maxf_90_r2 = round(Calculater(seg_mask2_3d[100:,:,label2], edge=True).maxf90,2)
            
            avg_tmt_minf = abs((objR_pred_minf_l1+objR_pred_minf_r1)/2 - (objR_pred_minf_l2+objR_pred_minf_r2)/2)
            avg_tmt_minf90 = abs((objR_pred_maxf_90_l1+objR_pred_maxf_90_r1)/2 - (objR_pred_maxf_90_l2+objR_pred_maxf_90_r2)/2)
            
            print("Slice diff", idx, slice_diff,dice_diff,iou_diff,
                  objR_pred_minf_l1,
                  objR_pred_maxf_90_l1,
                  objR_pred_minf_r1,
                  objR_pred_maxf_90_r1,
                  
                  objR_pred_minf_l2,
                  objR_pred_maxf_90_l2,
                  objR_pred_minf_r2,
                  objR_pred_maxf_90_r2,
                  avg_tmt_minf,
                  avg_tmt_minf90)
            
            results_list.append([patient_id, slice_diff,
                                 dice_diff1,dice_diff2,
                                 dice_diff,iou_diff,
                  objR_pred_minf_l1,
                  objR_pred_maxf_90_l1,
                  objR_pred_minf_r1,
                  objR_pred_maxf_90_r1,
                  
                  objR_pred_minf_l2,
                  objR_pred_maxf_90_l2,
                  objR_pred_minf_r2,
                  objR_pred_maxf_90_r2,
                  avg_tmt_minf,
                  avg_tmt_minf90])
        
            #break

    df=pd.DataFrame(np.asarray(results_list))
    df.to_csv(output_dir+"icc_dice1_dice2_healthy.csv", header=["ID", "slice_diff", 'dice_diff1','dice_diff2',
                                             "dice_diff","iou_diff",
                                             "TM1 minf a1",
                                             "TM1 90 a1",
                                             "TM2 minf a1",
                                             "TM2 90 a1",
                                             
                                             "TM1 minf a2",
                                             "TM1 90 a2",
                                             "TM2 minf a2",
                                             "TM2 90 a2",
                                             "avg_tmt_minf",
                                             "avg_tmt_minf90"])
    
              
# measure difference in segmentaion mask
icc_masks(metadata_1_path, annotator_1_healthy_path, annotator_2_healthy_path, output_dir)
#icc_masks(metadata_2_path, annotator_1_cancer_path, annotator_2_cancer_path, output_dir)
