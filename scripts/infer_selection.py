import sys
 
# setting path
sys.path.append('../TM2_segmentation')

from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np 
import pandas as pd
import tensorflow as tf
import glob, os, functools
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras import backend as K
import nibabel as nib

from scripts.densenet_regression import DenseNet
from scripts.preprocess_utils import brain_norm_masked, enhance,load_nii,get_id_and_path
#from preprocessing.masks_to_npy import get_id_and_path

def funcy(img):
    return img

def get_slice_number_from_prediction(predictions):
    predictions = 2.0 * (predictions - 0.5)  # sigmoid.output = True
    smoothing_kernel = 5.0 # number of MIP slices
    smoothed_predictions = gaussian_filter(predictions, smoothing_kernel)

    zero_crossings = []
    for s in range(len(smoothed_predictions) - 1):
        if (smoothed_predictions[s] < 0.0) != (smoothed_predictions[s + 1] < 0.0):
            if(abs(smoothed_predictions[s]) < abs(smoothed_predictions[s + 1])):
                zero_crossings.append(s)
            else:
                zero_crossings.append(s + 1)
    if len(zero_crossings) == 0:
        smoothed_predictions = abs(smoothed_predictions).flatten()
        indices = (smoothed_predictions).argsort()[:10]
        chosen_index = np.argmin(abs(smoothed_predictions))   
    else: 
        chosen_index = np.amax(zero_crossings) #sorted beforezero_crossings[:1]
    return chosen_index

def test(image_dir, model_weight_path, csv_write_path,input_annotation_file):
    #with tf.device('/gpu:0'):
    model = DenseNet(img_dim=(256, 256, 1), 
                nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                compression_rate=0.5, sigmoid_output_activation=True, 
                activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
    model.load_weights(model_weight_path)
    print('\n','\n','\n','loaded:' ,model_weight_path)  
    
    df_prediction = pd.DataFrame()
    df = pd.read_csv(input_annotation_file,header=0)

    df_to_be_writen = pd.DataFrame()
    df_train = df[df['train/test']=='test']
    df_train=df_train[df_train['Ok registered? Y/N']=='Y'].reset_index()

    mae = []    
    for idx in range(0, df_train.shape[0]):
        row = df_train.iloc[idx]
        patient_id, image_path, tm_file,_ = get_id_and_path(row, image_dir, True)
        if image_path != 0:
            image_sitk = sitk.ReadImage(image_path)    
            windowed_images  = sitk.GetArrayFromImage(image_sitk)
            
            seg_data, seg_affine = load_nii(tm_file)
            
            # find the slice on which we made annotations
            if (np.asarray(nib.aff2axcodes(seg_affine))==['R', 'A', 'S']).all():
                 # check the orientation you wanna reorient.
                # todo: create a function
                ornt = np.array([[0, -1],
                                [1, -1],
                                [2, 1]])
                
                seg = nib.load(tm_file)
                seg = seg.as_reoriented(ornt)
                seg_data, seg_affine  = seg.get_fdata(), seg.affine
                slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
                
            elif (np.asarray(nib.aff2axcodes(seg_affine))==['L', 'P', 'S']).all():
                # check the orientation you wanna reorient.
                # todo: create a function
                ornt = np.array([[0, -1],
                                [1, 1],
                                [2, 1]])
                
                seg = nib.load(tm_file)
                seg = seg.as_reoriented(ornt)
                seg_data, seg_affine  = seg.get_fdata(), seg.affine
                slice_label = np.asarray(np.where(seg_data != 0)).T[0, 2] 
                
            # enchancing is done in the preprocessing step
            resize_func = functools.partial(resize, output_shape=model.input_shape[1:3],
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

            predictions = model.predict(series_w)
            chosen_index = get_slice_number_from_prediction(predictions)
        
            df_inter = pd.DataFrame({'patient_id':image_path.split('/')[-1].split('.')[0],
                                    'Predict_slice':chosen_index,
                                    #'Z_spacing':round(image_sitk.GetSpacing()[-1],5),
                                    #'XY_spacing': round(image_sitk.GetSpacing()[0],5),
                                    'GT slice':slice_label},index=[0])
            df_prediction = df_prediction.append(df_inter)
            df_prediction.to_csv(csv_write_path)
            print(idx,' path: ',image_path,'\n','predicted slice:',chosen_index," GT slice:",slice_label,'\n')
            mae.append(abs(chosen_index-slice_label))
            #break
        
    print('Slice prediction is written into:',csv_write_path,'\n')
    print('AVG MAE ', np.average(mae))
