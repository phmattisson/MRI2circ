import sys
import os
 
# setting path
sys.path.append('../TM2_segmentation')

from scipy.ndimage.filters import gaussian_filter
from densenet_regression import DenseNet
from skimage.transform import resize,rescale
import SimpleITK as sitk
import numpy as np 
import pandas as pd
import tensorflow as tf
import glob, os, functools
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras import backend as K
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from numpy import median
import skimage

from unet import get_unet_2D
from preprocess_utils import  load_nii,save_nii ,crop_center
from settings import  target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, major_voting
from infer_segmentation import get_slice_number_from_prediction,funcy

# prediction of the segmentation
def predict_slice_and_segmentation(image_dir, model_weight_path_densenet,model_weight_path_unet):
    images = []
    for path in os.listdir(image_dir):
        if path == ".DS_Store":
            continue
        for img in os.listdir(image_dir+"/"+path):
            if ".nii" in img:
                images.append(image_dir+path+"/"+img)
    images = sorted(images)

    model_densenet = DenseNet(img_dim=(256, 256, 1), 
                nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                compression_rate=0.5, sigmoid_output_activation=True, 
                activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
    model_densenet.load_weights(model_weight_path_densenet)
    print('\n','\n','\n','loaded:' ,model_weight_path_densenet)  
    
    model_unet = get_unet_2D(unet_classes,(target_size_unet[0], target_size_unet[1], 1),\
        num_convs=2,  activation='relu',
        compression_channels=[16, 32, 64, 128, 256, 512],
        decompression_channels=[256, 128, 64, 32, 16])
    model_unet.load_weights(model_weight_path_unet)
    print('\n','\n','\n','loaded:' ,model_weight_path_unet)  

    for idx, image_path in enumerate(images):
        image_sitk = sitk.ReadImage(image_path)    
        image_array  = sitk.GetArrayFromImage(image_sitk)

        print(image_path)
        patient_id = image_path.split(".")[0].split("/")[-1]
        
        resize_func = functools.partial(resize, output_shape=model_densenet.input_shape[1:3],
                                        preserve_range=True, anti_aliasing=True, mode='constant')
        series = np.dstack([resize_func(im) for im in windowed_images])
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

        series_n = []

        for slice_idx in range(2,np.shape(series)[0]-2):
            im_array = np.zeros((256,256,1,5))
            
            im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.uint8)
            im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.uint8)
            im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.uint8)
            im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.uint8)
            im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.uint8)
            
            im_array= np.max(im_array, axis=3)
            
            series_n.append(im_array)
            series_w = np.dstack([funcy(im) for im in series_n])
            series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])
        
        predictions = model_densenet.predict(series_w)
        slice_label = get_slice_number_from_prediction(predictions)

        ## unet
        image_array, seg_affine,_ = load_nii(image_path)
        seg_affine[0,2]= -seg_affine[0,2]
        seg_affine[1,2]= -seg_affine[1,2]
        infer_seg_array_3d = np.zeros(image_array.shape)
        image_array = enhance_noN4(image_array)

        # padding for image to become 256x256
        padded_image = np.pad(image_array[:,:,slice_label],[[29,30],[11,12]],'constant',constant_values=0)
            
        image_array_2d = rescale(padded_image, 2).reshape(1,target_size_unet[0],target_size_unet[1],1) 
        img_half_1 = np.concatenate((image_array_2d[:,:256,:,:],np.zeros_like(image_array_2d[:,:256,:,:])),axis=1)
        img_half_2 = np.concatenate((np.zeros_like(image_array_2d[:,256:,:,:]),image_array_2d[:,256:,:,:]),axis=1)

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
            infer_seg_array = model_unet.predict(image)
            muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)                  
            list_of_left_muscle_preds.append(muscle_seg)
                    
        for image in list_of_right_muscle: 
            infer_seg_array = model_unet.predict(image)
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
                
        threshold = 0.5 # ie must be present on 3 out of 4 predictions
        left_half_result = np.mean(list_of_left_muscle_preds_halved, axis=0)>=threshold
        right_half_result = np.mean(list_of_right_muscle_preds_halved, axis=0)>=threshold

        muscle_seg = np.concatenate((left_half_result,right_half_result),axis=1)
        infer_seg_array_2d = rescale(muscle_seg[0],0.5)
        infer_seg_array_3d[:,:,slice_label] = crop_center(infer_seg_array_2d[:,:,0],233,197)

        # TODO: rename tm_path
        tm_path = image_dir+image_path.split("/")[-1].split(".")[0] + '/AI_TM.nii.gz'
        print(slice_label, tm_path)
        save_nii(infer_seg_array_3d, tm_path, seg_affine)
        # TODO: save result to image and write metrics to csv 

if __name__=="__main__":
    model_weight_path_densenet = "model/densenet_models/test/Top_Slice_Weights.hdf5"
    image_dir = "data/t1_mris/registered"
    model_weight_path_unet = 'model/unet_models/test/Top_Segmentation_Model_Weight.hdf5'
    output_csv = 'data/predicted.csv'

    predict_slice_and_segmentation(image_dir, model_weight_path_densenet,model_weight_path_unet)