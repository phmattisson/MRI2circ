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
from settings import target_size_unet, unet_classes,major_voting, softmax_threshold,scaling_factor
major_voting_flag=major_voting
from scripts.feret import Calculater
from compute_population_pred import  measure_tm, major_voting, filter_islands,compute_crop_line


def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width)
    batch_size = y_true.shape[0]
    mean_dice_channel = 0.
    for i in range(batch_size):
        channel_dice = single_dice_coef(y_true[i, :, :], y_pred_bin[i, :, :])
        mean_dice_channel += channel_dice/(batch_size)
    return mean_dice_channel

def median_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width)
    batch_size = y_true.shape[0]
    median_dice_channel = []
    for i in range(batch_size):
        channel_dice = single_dice_coef(y_true[i, :, :], y_pred_bin[i, :, :])
        median_dice_channel.append(channel_dice)
    t=sorted(median_dice_channel)   
    return median(t)

# compute dice coefficient
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

# get metadata : age and sex
def get_metadata(row, image_dir):
    patient_id, image_path, age, gender = "","","",""
    patient_id = str(row['Filename']).split(".")[0]

    try:
        gender =row['Sex']# row['SEX'] #row['Sex']
        age =  row['AGE_M'] 
    except:
        gender = row['SEX']
        age =  row['AGE_M'] 
    
    return age, gender, 

# test the model
def test(image_dir, model_weight_path, slice_csv_path, output_dir, measure_iou):
    model = get_unet_2D(unet_classes,(target_size_unet[0], target_size_unet[1], 1),\
        num_convs=2,  activation='relu',
        compression_channels=[16, 32, 64, 128, 256, 512],
        decompression_channels=[256, 128, 64, 32, 16])
    
    # load the model
    model.load_weights(model_weight_path)
    # load the csv file
    df_prediction = pd.read_csv(slice_csv_path,index_col=0)

    split_name = 'test'
    alpha = 0.8 
    df_prediction = df_prediction[df_prediction['train/test']==split_name]
    df_prediction=df_prediction[df_prediction['Ok registered? Y/N']=='Y'].reset_index()
   
    list_true_1,list_true_2, list_true = [],[],[]
    list_pred_1,list_pred_2, list_pred = [],[],[]
    list_csa = []
    lst_dices = []
    print("Testing n=",len(df_prediction))
    
    # loop over the csv file
    for idx, row in df_prediction.iterrows():
        print(idx)
        patient_id, image_path, tm_file,_ = get_id_and_path(row, image_dir, True)
        age, gender = get_metadata(row, image_dir)
        
        print(patient_id, image_path, tm_file)
        if image_path != 0:
            seg_data, seg_affine = load_nii(tm_file)
            
            # check the orientation of the image
            if (np.asarray(nib.aff2axcodes(seg_affine))==['L', 'P', 'S']).all() or\
                    (np.asarray(nib.aff2axcodes(seg_affine))==['R', 'A', 'S']).all():
                slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
                image_array, affine = load_nii(image_path)
            ''' elif (np.asarray(nib.aff2axcodes(seg_affine))==['L', 'P', 'S']).all():
                # check the orientation you wanna reorient.
                # todo: create a function
                ornt = np.array([[0, -1],
                                [1, 1],
                                [2, 1]])
                
                seg = nib.load(tm_file)
                img = nib.load(image_path)
                
                seg = seg.as_reoriented(ornt)
                img = img.as_reoriented(ornt)
                
                image_array, affine = img.get_fdata(), img.affine
                seg_data, seg_affine  = seg.get_fdata(), seg.affine
                
                slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
            '''
            # create empty arrays        
            infer_seg_array_3d_1,infer_seg_array_3d_2 = np.zeros(image_array.shape),np.zeros(image_array.shape)
            infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered = np.zeros(image_array.shape),np.zeros(image_array.shape)
            infer_seg_array_3d_merged_filtered =  np.zeros(image_array.shape)
            
            # rescale the image
            image_array_2d = rescale(image_array[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
            # split the image in two
            img_half_1 = np.concatenate((image_array_2d[:,:256,:,:],np.zeros_like(image_array_2d[:,:256,:,:])),axis=1)
            img_half_2 = np.concatenate((np.zeros_like(image_array_2d[:,256:,:,:]),image_array_2d[:,256:,:,:]),axis=1)

            # predict the segmentation and filter the segmentation
            if major_voting_flag ==False:
                infer_seg_array_1 = model.predict(img_half_1)
                infer_seg_array_2 = model.predict(img_half_2)

                muscle_seg_1 = (infer_seg_array_1[:,:,:,1] >= softmax_threshold).reshape(1,target_size_unet[0],target_size_unet[1],1)   #<>          
                muscle_seg_2 = (infer_seg_array_2[:,:,:,1] >= softmax_threshold).reshape(1,target_size_unet[0],target_size_unet[1],1)   #<>       
            else:
                muscle_seg_1, muscle_seg_2 = major_voting(image_array_2d, model)

            # plot the segmentation
            fg=plt.figure(figsize=(5, 5), facecolor='k')
            plt.imshow(image_array_2d[0],'gray')
            plt.imshow(muscle_seg_1[0], 'gray', alpha=0.4, interpolation='none')
            #plt.savefig(output_dir+"/pics/"+patient_id+"_1.png")

            fg=plt.figure(figsize=(5, 5), facecolor='k')
            plt.imshow(image_array_2d[0],'gray')
            plt.imshow(muscle_seg_2[0], 'gray', alpha=0.4, interpolation='none')
            #plt.savefig(output_dir+"/pics/"+patient_id+"_2.png")
            
            # filter islands
            muscle_seg_1_filtered, area_1, cnt_1 = filter_islands(muscle_seg_1[0])
            muscle_seg_2_filtered, area_2, cnt_2 = filter_islands(muscle_seg_2[0])

            ## save plots 
            fg = plt.figure(figsize=(5, 5), facecolor='k')
            I = cv2.normalize(image_array_2d[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(output_dir+"/pics/"+"_"+patient_id+".png", I)
            im = cv2.imread(output_dir+"/pics/"+"_"+patient_id+".png")                        
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

            cv2.imwrite(output_dir+"/pics/"+"_"+patient_id+"_mask.png", result)
            
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
            infer_3d_path = output_dir+"/pics/"+patient_id + '_AI_seg.nii.gz'
            #save_nii(infer_seg_array_3d_merged_filtered, infer_3d_path, affine)
                
            # measure TM
            if measure_iou:
                objL_gt_minf = round(Calculater(seg_data[:100,:,slice_label], edge=True).minf,2)
                objR_gt_minf = round(Calculater(seg_data[100:,:,slice_label], edge=True).minf,2)
                
                print("Dice score TM1", round(dice_coef(seg_data[:100,:,slice_label],infer_seg_array_3d_1_filtered[:100,:,slice_label]),2))
                print("Dice score TM2", round(dice_coef(seg_data[100:,:,slice_label],infer_seg_array_3d_2_filtered[100:,:,slice_label]),2))
                
                lst_dices.append([patient_id, 
                                 gender,
                                 age, 
                                 slice_label,
                                 round(dice_coef(seg_data[:100,:,slice_label],infer_seg_array_3d_1_filtered[:100,:,slice_label]),2),
                                 round(dice_coef(seg_data[100:,:,slice_label],infer_seg_array_3d_2_filtered[100:,:,slice_label]),2)])
                
                print("CSA GT TM1 vs Pred", np.sum(seg_data[:100,:,slice_label]), np.sum(infer_seg_array_3d_1_filtered[:100,:,slice_label]))
                print("CSA GT TM2 vs Pred", np.sum(seg_data[100:,:,slice_label]), np.sum(infer_seg_array_3d_2_filtered[100:,:,slice_label]))
                
                crop_line = compute_crop_line(image_array[:,15:-21,slice_label],infer_seg_array_2d_1,infer_seg_array_2d_2)
                print(np.shape(infer_seg_array_3d_1),np.shape(infer_seg_array_3d_1_filtered))
                unfiltered_metrics = measure_tm(image_array,infer_seg_array_3d_1,infer_seg_array_3d_2,slice_label,crop_line)
                filtered_metrics = measure_tm(image_array,infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered,slice_label,crop_line)
                gt_metrics =  measure_tm(image_array,seg_data,seg_data,slice_label,crop_line)
                
                list_csa.append([patient_id, 
                                 gender,
                                 age, 
                                 slice_label,
                                 *gt_metrics,
                                 *unfiltered_metrics, 
                                 *filtered_metrics])
                
                list_true.append(np.concatenate((seg_data[:100,:,slice_label],seg_data[100:,:,slice_label]),axis=0))
                list_pred.append(np.concatenate((infer_seg_array_3d_1_filtered[:100,:,slice_label],infer_seg_array_3d_2_filtered[100:,:,slice_label]),axis=0))
                
            gc.collect()      
        
    print(np.shape(np.asarray(list_true)[:,:,int(crop_line):]))
    print("Mean dice TM:", round(mean_dice_coef(np.asarray(list_true)[:,:,int(crop_line):],np.asarray(list_pred)[:,:,int(crop_line):]),3))
    print("Median dice TM :",round(median_dice_coef(np.asarray(list_true)[:,:,int(crop_line):],np.asarray(list_pred)[:,:,int(crop_line):]),3))
    
    df=pd.DataFrame(np.asarray(list_csa))
    df_dices=pd.DataFrame(np.asarray(lst_dices))
    df_dices.to_csv("data/dices.csv", header=["ID","Gender", "Age", "Slice label",
                                              'Dice1','Dice2'])
    
    df.to_csv(output_dir+"csa_health.csv", header=["ID","Gender", "Age", "Slice label",
                                            ## GT
                                            "CSA GT TM1","CSA GT TM2",
                                            "CSA GT SUM", 
                                            "CSA GT AVG",
                                            "TM GT1","TM GT2",
                                            "TMT GT SUM", 
                                            "TMT GT AVG", 
                                            "TM GT1 90","TM GT2 90",
                                            "TMT GT SUM 90", 
                                            "TMT GT AVG 90", 
                                            "CSA GT TM1 w line", "CSA GT TM2 w line",
                                            "CSA GT SUM w line",
                                            "CSA GT AVG w line",
                                            "TM GT1 w line","TM GT2 w line",
                                            "TMT GT SUM w line",
                                            "TMT GT AVG w line",
                                            ##unfiltered
                                            "CSA PRED TM1", "CSA PRED TM2",
                                            "CSA PRED SUM", 
                                            "CSA PRED AVG",
                                            "TM PRED1","TM PRED2",
                                            "TMT PRED SUM", 
                                            "TMT PRED AVG", 
                                            "TM PRED1 90","TM PRED2 90",
                                            "TMT PRED SUM 90", 
                                            "TMT PRED AVG 90", 
                                            "CSA PRED TM1 w line", "CSA PRED TM2 w line",
                                            "CSA PRED SUM w line",
                                            "CSA PRED AVG w line",
                                            "TM PRED1 w line","TM PRED2 w line",
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
                                            "CSA PRED TM1 w line filtered", "CSA PRED TM2 w line filtered",
                                            "CSA PRED SUM w line filtered",
                                            "CSA PRED AVG w line filtered",
                                            "TM PRED1 w line filtered","TM PRED2 w line filtered",
                                            "TMT PRED SUM w line filtered",
                                            "TMT PRED AVG w line filtered"
                                             ])



