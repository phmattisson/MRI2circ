import os
import argparse
import numpy as np
import sys
sys.path.append('../')
import SimpleITK as sitk
import nibabel as nib
import tensorflow as tf
import pandas as pd
from scripts.densenet_regression import DenseNet
from scripts.preprocess_utils import load_nii, enhance_noN4
from scripts.infer_selection import get_slice_number_from_prediction, funcy
from skimage.transform import resize
import functools
import ants
import matplotlib.pyplot as plt
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.zscore import ZScoreNormalize
import subprocess
import matplotlib
matplotlib.use('Agg')
import argparse
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# Import necessary functions from mips.py
from mips import process_and_visualize, ThresholdFilter,length_of_contour_with_spacing,distance_2d_with_spacing,distance_2d_with_spacing

def register_images(fixed_image_path, moving_image_path):
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')
    return registration['warpedmovout']

def select_template_based_on_age(age, neonatal):
    if neonatal:
        if age == 36:
            golden_file_path = "../shared_data/mni_templates/mean/ga_36/template_t1.nii.gz"
        elif age == 37:
            golden_file_path = "../shared_data/mni_templates/mean/ga_37/template_t1.nii.gz"
        elif age == 38:
            golden_file_path = "../shared_data/mni_templates/mean/ga_38/template_t1.nii.gz"
        elif age == 39:
            golden_file_path = "../shared_data/mni_templates/mean/ga_39/template_t1.nii.gz"
        elif age == 40:
            golden_file_path = "../shared_data/mni_templates/mean/ga_40/template_t1.nii.gz"
        elif age == 41:
            golden_file_path = "../shared_data/mni_templates/mean/ga_41/template_t1.nii.gz"
        elif age == 42:
            golden_file_path = "../shared_data/mni_templates/mean/ga_42/template_t1.nii.gz"
        elif age == 43:
            golden_file_path = "../shared_data/mni_templates/mean/ga_43/template_t1.nii.gz"
        elif age == 44:
            golden_file_path = "../shared_data/mni_templates/mean/ga_44/template_t1.nii.gz"
        return golden_file_path
    else:
        age_ranges = {
            "../shared_data/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "../shared_data/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "../shared_data/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}
        for golden_file_path, age_values in age_ranges.items():
            if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
                print(golden_file_path)
                return golden_file_path


def zscore_normalize(input_file, output_file):
    img = nib.load(input_file)
    data = img.get_fdata()

    mean = np.mean(data)
    std = np.std(data)
    z_scored = (data - mean) / std

    new_img = nib.Nifti1Image(z_scored, img.affine, img.header)

    nib.save(new_img, output_file)

def main(img_path, age, output_path, neonatal, theta_x=0, theta_y=0, theta_z=0, 
         conductance_parameter=3.0, smoothing_iterations=5, time_step=0.0625, 
         threshold_filter=ThresholdFilter.Otsu, mip_slices=5):
    # Select template
    template_path = select_template_based_on_age(age, neonatal)

    # Register image to template
    registered_image = register_images(template_path, img_path)
    
    # Save the registered image (without enhancement)
    patient_id = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(output_path, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    registered_path = os.path.join(output_dir, 'registered.nii.gz')
    ants.image_write(registered_image, registered_path)
    
    if not os.path.exists(output_dir + "/no_z"):
        os.mkdir(output_dir + "/no_z")
    
    # Convert to numpy array and enhance for slice selection
    image_sitk = sitk.ReadImage(registered_path)
    image_array = sitk.GetArrayFromImage(image_sitk)
    enhanced_image_array = enhance_noN4(image_array)
    image3 = sitk.GetImageFromArray(enhanced_image_array)
    
    sitk.WriteImage(image3, output_dir + "/no_z/registered_no_z.nii")
    ''' 
    cmd_line = f"zscore-normalize {output_dir}/no_z/registered_no_z.nii -o {output_dir}/registered_z.nii"
    subprocess.getoutput(cmd_line)     
    print(cmd_line)
    '''
    
    input_file = f"{output_dir}/no_z/registered_no_z.nii"
    output_file = f"{output_dir}/registered_z.nii"
    zscore_normalize(input_file, output_file)
    print("Preprocessing done!")
    image_sitk = sitk.ReadImage(output_dir + '/registered_z.nii')    
    enhanced_image_array = sitk.GetArrayFromImage(image_sitk)  

    # Slice selection (using enhanced image)
    model_selection = DenseNet(img_dim=(256, 256, 1), 
                    nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                    compression_rate=0.5, sigmoid_output_activation=True, 
                    activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True)
    model_selection.load_weights('../model/densenet_models/test/itmt1.hdf5')

    resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                    preserve_range=True, anti_aliasing=True, mode='constant')
    series = np.dstack([resize_func(im) for im in enhanced_image_array])
    series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
    series_n = []

    for slice_idx in range(2, np.shape(series)[0]-2):
        im_array = np.zeros((256, 256, 1, 5))
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
    print("Predicted slice:", slice_label)
    print(f'registered path: {registered_path}')
    # Calculate circumference using mips.py function (on non-enhanced registered image)
    # After parsing arguments
    if args.threshold_filter == "Otsu":
        threshold_filter_enum = ThresholdFilter.Otsu
    elif args.threshold_filter == "Binary":
        threshold_filter_enum = ThresholdFilter.Binary
    else:
        raise ValueError(f"Invalid threshold filter: {args.threshold_filter}")

    
    circumference, contour_array, mip_array = process_and_visualize(
        registered_path,
        slice_num=slice_label,
        theta_x=theta_x,
        theta_y=theta_y,
        theta_z=theta_z,
        conductance_parameter=conductance_parameter,
        smoothing_iterations=smoothing_iterations,
        time_step=time_step,
        threshold_filter=threshold_filter_enum,
        mip_slices=mip_slices
    )

    # Save results
    df_results = pd.DataFrame({
        'patient_id': [patient_id],
        'circumference': [circumference]
    })
    df_results.to_csv(os.path.join(output_dir, 'circumference.csv'), index=False)

    # Save the visualization
    plt.ioff()
    plt.figure(figsize=(10, 10))
    plt.imshow(mip_array, cmap='gray')
    plt.contour(contour_array, colors='r')
    plt.title(f"Head Circumference: {circumference:.2f} mm")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'contour_visualization.png'))
    plt.close()

    print(f"Circumference: {circumference:.2f} mm")
    return circumference
'''
if __name__ == "__main__":    
    path = '/home/philip-mattisson/Desktop/data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz'
    age = 35
    output = '/home/philip-mattisson/Desktop/data/V2Out'

    result = main(path, age, output,False)
    if result:
        print(f"Calculated Head Circumference: {result:.2f} mm")
    else:
        print("Failed to process image.")
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate brain perimeter from MRI image.")
    parser.add_argument("img_path", type=str, help="Path to the MRI image file")
    parser.add_argument("age", type=int, help="Age of the subject")
    parser.add_argument("output_path", type=str, help="Path to the output folder")
    parser.add_argument("--neonatal", action="store_true", help="Flag to indicate if the subject is neonatal")
    parser.add_argument("--theta_x", type=float, default=0, help="Rotation angle around x-axis")
    parser.add_argument("--theta_y", type=float, default=0, help="Rotation angle around y-axis")
    parser.add_argument("--theta_z", type=float, default=0, help="Rotation angle around z-axis")
    parser.add_argument("--conductance_parameter", type=float, default=3.0, help="Conductance parameter for anisotropic diffusion")
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="Number of smoothing iterations")
    parser.add_argument("--time_step", type=float, default=0.0625, help="Time step for anisotropic diffusion")
    parser.add_argument("--threshold_filter", type=str, default="Otsu", choices=["Otsu", "Binary"], help="Threshold filter method")
    parser.add_argument("--mip_slices", type=int, default=5, help="Number of slices for maximum intensity projection")
    
    args = parser.parse_args()
    
    result = main(args.img_path, args.age, args.output_path, args.neonatal,
                  theta_x=args.theta_x, theta_y=args.theta_y, theta_z=args.theta_z,
                  conductance_parameter=args.conductance_parameter,
                  smoothing_iterations=args.smoothing_iterations,
                  time_step=args.time_step,
                  threshold_filter=args.threshold_filter,
                  mip_slices=args.mip_slices)
    
    if result:
        print(f'circumference {result}')
    else:
        print("Failed to process image.")

