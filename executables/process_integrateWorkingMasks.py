import os
import argparse
from re import I
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
from shapely.geometry import Polygon,LineString,MultiPolygon,GeometryCollection
from shapely.ops import split
from shapely.affinity import scale
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
# Import necessary functions from mips.py
from mips import process_and_visualize, ThresholdFilter,length_of_contour_with_spacing,distance_2d_with_spacing,distance_2d_with_spacing
import plotly.graph_objects as go

def register_images(fixed_image_path, moving_image_path):
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')
    return registration['warpedmovout']

def select_template_based_on_age(age, neonatal,months):
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
    elif months:
        if age == 0:
            golden_file_path = "../shared_data/mni_templates/months/0Month/BCP-00M-T1.nii.gz"
        elif age == 1:
            golden_file_path = "../shared_data/mni_templates/months/1Month/BCP-01M-T1.nii.gz"
        elif age == 2:
            golden_file_path = "../shared_data/mni_templates/months/2Month/BCP-02M-T1.nii.gz"
        elif age == 3:
            golden_file_path = "../shared_data/mni_templates/months/3Month/BCP-03M-T1.nii.gz"
        elif age == 4:
            golden_file_path = "../shared_data/mni_templates/months/4Month/BCP-04M-T1.nii.gz"
        elif age == 5:
            golden_file_path = "../shared_data/mni_templates/months/5Month/BCP-05M-T1.nii.gz"
        elif age == 6:
            golden_file_path = "../shared_data/mni_templates/months/6Month/BCP-06M-T1.nii.gz"
        elif age == 7:
            golden_file_path = "../shared_data/mni_templates/months/7Month/BCP-07M-T1.nii.gz"
        elif age == 8:
            golden_file_path = "../shared_data/mni_templates/months/8Month/BCP-08M-T1.nii.gz"
        elif age == 9:
            golden_file_path = "../shared_data/mni_templates/months/9Month/BCP-09M-T1.nii.gz"
        elif age == 10:
            golden_file_path = "../shared_data/mni_templates/months/10Month/BCP-10M-T1.nii.gz"
        elif age == 11:
            golden_file_path = "../shared_data/mni_templates/months/11Month/BCP-11M-T1.nii.gz"
        elif age == 12:
            golden_file_path = "../shared_data/mni_templates/months/12Month/BCP-12M-T1.nii.gz"
        elif age == 13:
            golden_file_path = "../shared_data/mni_templates/months/13Month/BCP-13M-T1.nii.gz"
        elif age == 14:
            golden_file_path = "../shared_data/mni_templates/months/14Month/BCP-14M-T1.nii.gz"
        elif age == 15:
            golden_file_path = "../shared_data/mni_templates/months/15Month/BCP-15M-T1.nii.gz"
        elif age == 16:
            golden_file_path = "../shared_data/mni_templates/months/16Month/BCP-16M-T1.nii.gz"
        elif age == 17:
            golden_file_path = "../shared_data/mni_templates/months/17Month/BCP-17M-T1.nii.gz"
        elif age == 18:
            golden_file_path = "../shared_data/mni_templates/months/18Month/BCP-18M-T1.nii.gz"
        elif age == 19:
            golden_file_path = "../shared_data/mni_templates/months/19Month/BCP-19M-T1.nii.gz"
        elif age == 20:
            golden_file_path = "../shared_data/mni_templates/months/20Month/BCP-20M-T1.nii.gz"
        elif age == 21:
            golden_file_path = "../shared_data/mni_templates/months/21Month/BCP-21M-T1.nii.gz"
        elif age == 22:
            golden_file_path = "../shared_data/mni_templates/months/22Month/BCP-22M-T1.nii.gz"
        elif age == 23:
            golden_file_path = "../shared_data/mni_templates/months/23Month/BCP-23M-T1.nii.gz"
        elif age == 24:
            golden_file_path = "../shared_data/mni_templates/months/24Month/BCP-24M-T1.nii.gz"
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

def main(img_path, age, output_path, neonatal, months, ranges, theta_x=0, theta_y=0, theta_z=0, 
         conductance_parameter=3.0, smoothing_iterations=5, time_step=0.0625, 
         threshold_filter=ThresholdFilter.Otsu, mip_slices=5):
    # Select template
    template_path = select_template_based_on_age(age, neonatal, months)

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

    # Ensure 'funcy' is defined or replace it with appropriate preprocessing
    # For now, let's assume 'funcy' is an identity function
    series_w = np.dstack([im for im in series_n])
    series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])

    predictions = model_selection.predict(series_w)
    slice_label = get_slice_number_from_prediction(predictions)
    print(slice_label)

    # Determine threshold filter
    if threshold_filter == "Otsu":
        threshold_filter_enum = ThresholdFilter.Otsu
    elif threshold_filter == "Binary":
        threshold_filter_enum = ThresholdFilter.Binary
    else:
        raise ValueError(f"Invalid threshold filter: {threshold_filter}")

    if ranges:
        slice_interval = list(range(88, 108))
        results = []
        for slice_range in slice_interval:
            circumference, contour_array, mip_array, spacing, largest_component_array = process_and_visualize(
                registered_path,
                slice_num=slice_range,
                theta_x=theta_x,
                theta_y=theta_y,
                theta_z=theta_z,
                conductance_parameter=conductance_parameter,
                smoothing_iterations=smoothing_iterations,
                time_step=time_step,
                threshold_filter=threshold_filter_enum,
                mip_slices=mip_slices
            )
            results.append(circumference)

    # Initialize lists to store all measurements and masks
    all_measurements = []
    volumes = {
        'posterior_left': 0,
        'posterior_right': 0,
        'anterior_left': 0,
        'anterior_right': 0,
        'total': 0
    }
    volume_masks = []
    quadrant_masks = {
        'posterior_left': [],
        'posterior_right': [],
        'anterior_left': [],
        'anterior_right': []
    }

    # Process slices until circumference condition is met
    circumference = 1000
    k = 0
    while circumference > 100:
        try:
            circumference, contour_array, mip_array, spacing, largest_component_array = process_and_visualize(
                registered_path,
                slice_num=slice_label + k,
                theta_x=theta_x,
                theta_y=theta_y,
                theta_z=theta_z,
                conductance_parameter=conductance_parameter,
                smoothing_iterations=smoothing_iterations,
                time_step=time_step,
                threshold_filter=threshold_filter_enum,
                mip_slices=mip_slices
            )
        except Exception as e:
            print(f'Error with circumference calculation: {e}')
            df_results = pd.DataFrame(all_measurements)
            summary_stats = {
                'patient_id': patient_id,
                'circumference_median': df_results['circumference_mm'].median(),
                'circumference_mean': df_results['circumference_mm'].mean(),
                'circumference_std': df_results['circumference_mm'].std(),
                'area_median': df_results['area_cm2'].median(),
                'area_mean': df_results['area_cm2'].mean(),
                'area_std': df_results['area_cm2'].std()
            }
            # Save detailed results
            df_results.to_csv(os.path.join(output_dir, 'detailed_measurements.csv'), index=False)

            # Save summary statistics
            pd.DataFrame([summary_stats]).to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
            return circumference

        k += 1

        # Calculate measurements
        y_indices, x_indices = np.where(contour_array > 0)

        # Calculate bounding box
        minx_idx = np.min(x_indices)
        maxx_idx = np.max(x_indices)
        miny_idx = np.min(y_indices)
        maxy_idx = np.max(y_indices)

        # Calculate dimensions
        width_mm = (maxx_idx - minx_idx) * spacing[0]
        length_mm = (maxy_idx - miny_idx) * spacing[1]
        center_x_idx = (minx_idx + maxx_idx) // 2
        center_y_idx = (miny_idx + maxy_idx) // 2

        # Calculate areas and quadrants
        height, width = largest_component_array.shape
        Y, X = np.ogrid[:height, :width]

        upper_mask = Y < center_y_idx
        lower_mask = Y >= center_y_idx
        left_mask = X < center_x_idx
        right_mask = X >= center_x_idx

        upper_left_mask = upper_mask & left_mask
        upper_right_mask = upper_mask & right_mask
        lower_left_mask = lower_mask & left_mask
        lower_right_mask = lower_mask & right_mask

        # Calculate quadrant areas
        area_upper_left = np.sum(largest_component_array & upper_left_mask) * spacing[0] * spacing[1] / 100
        area_upper_right = np.sum(largest_component_array & upper_right_mask) * spacing[0] * spacing[1] / 100
        area_lower_left = np.sum(largest_component_array & lower_left_mask) * spacing[0] * spacing[1] / 100
        area_lower_right = np.sum(largest_component_array & lower_right_mask) * spacing[0] * spacing[1] / 100
        area_pixels = np.sum(largest_component_array)

        z_spacing_cm = spacing[2] / 10

        # Convert area to square millimeters
        area_mm2 = area_pixels * spacing[0] * spacing[1]
        area_cm2 = area_mm2 / 100

        # Store the quadrant areas in a dictionary
        quadrant_areas = {
            'posterior_left': area_upper_left,
            'posterior_right': area_upper_right,
            'anterior_left': area_lower_left,
            'anterior_right': area_lower_right
        }

        # Calculate volumes for this slice
        slice_contribution = {
            'posterior_left': area_upper_left * z_spacing_cm,
            'posterior_right': area_upper_right * z_spacing_cm,
            'anterior_left': area_lower_left * z_spacing_cm,
            'anterior_right': area_lower_right * z_spacing_cm,
            'total': area_cm2 * z_spacing_cm
        }

        # Add this slice's contribution to total volumes
        for key in volumes:
            volumes[key] += slice_contribution[key]

        # Append the largest component array to volume masks
        volume_masks.append(largest_component_array)

        # Append quadrant masks
        quadrant_masks['posterior_left'].append(largest_component_array & upper_left_mask)
        quadrant_masks['posterior_right'].append(largest_component_array & upper_right_mask)
        quadrant_masks['anterior_left'].append(largest_component_array & lower_left_mask)
        quadrant_masks['anterior_right'].append(largest_component_array & lower_right_mask)

        # (Visualization code remains the same)

        # Store measurements
        measurements = {
            'patient_id': patient_id,
            'iteration': k,
            'slice_number': slice_label + k,
            'circumference_mm': circumference,
            'width_mm': width_mm,
            'length_mm': length_mm,
            'area_cm2': area_cm2,
            'cephalic_index': (width_mm / length_mm) * 100,
            'posterior_left_area_cm2': area_upper_left,
            'posterior_right_area_cm2': area_upper_right,
            'anterior_left_area_cm2': area_lower_left,
            'anterior_right_area_cm2': area_lower_right,
            # Add accumulated volumes up to this slice
            'total_volume_cm3': volumes['total'],
            'posterior_left_volume_cm3': volumes['posterior_left'],
            'posterior_right_volume_cm3': volumes['posterior_right'],
            'anterior_left_volume_cm3': volumes['anterior_left'],
            'anterior_right_volume_cm3': volumes['anterior_right'],
            # Add this slice's volume contribution
            'slice_volume_cm3': slice_contribution['total'],
            'slice_posterior_left_volume_cm3': slice_contribution['posterior_left'],
            'slice_posterior_right_volume_cm3': slice_contribution['posterior_right'],
            'slice_anterior_left_volume_cm3': slice_contribution['anterior_left'],
            'slice_anterior_right_volume_cm3': slice_contribution['anterior_right']
        }
        all_measurements.append(measurements)

# After processing all slices, create 3D visualization
  # After processing all slices, create 3D visualization
    if volume_masks:
        # Stack the volume masks to create a 3D array
        volume_mask_3d = np.stack(volume_masks, axis=0)  # Shape: (k, H, W)

        # Stack quadrant masks
        quadrant_masks_3d = {}
        for quadrant in quadrant_masks:
            quadrant_masks_3d[quadrant] = np.stack(quadrant_masks[quadrant], axis=0)  # Shape: (k, H, W)

        # Create a combined 3D array where each voxel is labeled according to its quadrant
        quadrant_volume = np.zeros_like(volume_mask_3d, dtype=np.uint8)

        quadrant_volume[quadrant_masks_3d['posterior_left'] > 0] = 1
        quadrant_volume[quadrant_masks_3d['posterior_right'] > 0] = 2
        quadrant_volume[quadrant_masks_3d['anterior_left'] > 0] = 3
        quadrant_volume[quadrant_masks_3d['anterior_right'] > 0] = 4

        # Now, create a full volume array of the same shape as mri_volume
        mri_volume = image_array  # Use the full MRI volume data without normalization

        mri_shape = mri_volume.shape  # Shape: (total_slices, H, W)
        quadrant_volume_full = np.zeros(mri_shape, dtype=np.uint8)  # Initialize full-sized quadrant volume

        # Insert the quadrant_volume into quadrant_volume_full at the correct slice positions
        start_slice = slice_label
        end_slice = slice_label + k

        # Ensure we don't exceed the MRI volume dimensions
        end_slice = min(end_slice, mri_shape[0])
        quadrant_slices = quadrant_volume[:end_slice - start_slice]

        quadrant_volume_full[start_slice:end_slice, :, :] = quadrant_slices

        # Ensure that quadrant_volume_full has non-zero values
        if np.count_nonzero(quadrant_volume_full) == 0:
            print("No voxels found in the quadrant_volume_full array.")
        else:

            fig = go.Figure()

            # Create coordinate grids
            z_dim, y_dim, x_dim = mri_volume.shape
            x_grid, y_grid, z_grid = np.meshgrid(
                np.arange(x_dim),
                np.arange(y_dim),
                np.arange(z_dim),
                indexing='ij'
            )

            # Flatten the grids and MRI volume for plotting
            x_flat = x_grid.flatten()
            y_flat = y_grid.flatten()
            z_flat = z_grid.flatten()
            mri_flat = mri_volume.flatten()

            # Plot the MRI data as a volume
            fig.add_trace(go.Volume(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                value=mri_flat,
                opacity=0.1,  # Adjust for desired transparency
                surface_count=15,
                colorscale='Gray',
                name='MRI Data',
                showscale=False
            ))

            # Overlay the quadrant masks
            '''
            for quadrant_label, color in zip([1, 2, 3, 4], ['red', 'green', 'blue', 'yellow']):
                mask = (quadrant_volume_full == quadrant_label)
                z_mask, y_mask, x_mask = np.nonzero(mask)
                fig.add_trace(go.Scatter3d(
                    x=x_mask,
                    y=y_mask,
                    z=z_mask,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=color,
                        opacity=0.5,
                    ),
                    name=f'Quadrant {quadrant_label}'
                ))
            '''

            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X', showbackground=False),
                    yaxis=dict(title='Y', showbackground=False),
                    zaxis=dict(title='Z', showbackground=False),
                    aspectmode='data'
                ),
                title='3D Volume Rendering of MRI with Quadrants',
                legend=dict(
                    x=0.8,
                    y=0.9
                )
            )

            # Save the figure
            fig.write_html(os.path.join(output_dir, '3D_volume_rendering.html'))
            print("3D visualization saved as '3D_volume_rendering.html'.")
            """
            # After processing all slices, create 3D visualization
        if volume_masks:
        # Stack the volume masks to create a 3D array
        volume_mask_3d = np.stack(volume_masks, axis=0)

        # Stack quadrant masks
        quadrant_masks_3d = {}
        for quadrant in quadrant_masks:
            quadrant_masks_3d[quadrant] = np.stack(quadrant_masks[quadrant], axis=0)

        # Create a combined 3D array where each voxel is labeled according to its quadrant
        quadrant_volume = np.zeros_like(volume_mask_3d, dtype=np.uint8)

        quadrant_volume[quadrant_masks_3d['posterior_left'] > 0] = 1
        quadrant_volume[quadrant_masks_3d['posterior_right'] > 0] = 2
        quadrant_volume[quadrant_masks_3d['anterior_left'] > 0] = 3
        quadrant_volume[quadrant_masks_3d['anterior_right'] > 0] = 4

        # Ensure that quadrant_volume has non-zero values
    if np.count_nonzero(quadrant_volume) == 0:
        print("No voxels found in the quadrant_volume array.")
    else:
        # Normalize MRI data for visualization
        mri_volume = image_array.astype(np.float32)
        mri_volume = (mri_volume - mri_volume.min()) / (mri_volume.max() - mri_volume.min())

        # Crop MRI data to match the dimensions of quadrant_volume
        mri_volume_cropped = mri_volume[slice_label:slice_label + k, :, :]

        # Verify dimensions match
        if mri_volume_cropped.shape != quadrant_volume.shape:
            print("Dimension mismatch between MRI data and quadrant volume.")
        else:
            # Create the MRI volume plot

            fig = go.Figure()

            # Create coordinate grids
            z_dim, y_dim, x_dim = quadrant_volume.shape
            x_grid = np.arange(x_dim)
            y_grid = np.arange(y_dim)
            z_grid = np.arange(z_dim)

            # Plot the MRI data as a volume
            fig.add_trace(go.Volume(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                value=mri_volume_cropped.flatten(),
                opacity=0.1,  # Adjust for desired transparency
                surface_count=10,
                colorscale='Gray',
                name='MRI Data'
            ))

            # Overlay the quadrant masks
            for quadrant_label, color in zip([1, 2, 3, 4], ['red', 'green', 'blue', 'yellow']):
                mask = (quadrant_volume == quadrant_label)
                z_mask, y_mask, x_mask = np.nonzero(mask)
                fig.add_trace(go.Scatter3d(
                    x=x_mask,
                    y=y_mask,
                    z=z_mask,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=color,
                        opacity=0.5,
                    ),
                    name=f'Quadrant {quadrant_label}'
                ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                title='3D Volume Rendering of MRI with Quadrants'
            )

            # Save the figure
            fig.write_html(os.path.join(output_dir, '3D_volume_rendering.html'))
            print("3D visualization saved as '3D_volume_rendering.html'.")
            """
 


    # Create DataFrame and save results
    df_results = pd.DataFrame(all_measurements)
    summary_stats = {
        'patient_id': patient_id,
        'circumference_median': df_results['circumference_mm'].median(),
        'circumference_mean': df_results['circumference_mm'].mean(),
        'circumference_std': df_results['circumference_mm'].std(),
        'area_median': df_results['area_cm2'].median(),
        'area_mean': df_results['area_cm2'].mean(),
        'area_std': df_results['area_cm2'].std()
    }

    # Save detailed results
    df_results.to_csv(os.path.join(output_dir, 'detailed_measurements.csv'), index=False)

    # Save summary statistics
    pd.DataFrame([summary_stats]).to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
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
    parser.add_argument("--months", action="store_true", help="Flag to indicate if age is specified in months")
    parser.add_argument("--ranges", action = "store_true", help='flag to indicate use of ranges instead of slice picker') 
    parser.add_argument("--theta_x", type=float, default=0, help="Rotation angle around x-axis")
    parser.add_argument("--theta_y", type=float, default=0, help="Rotation angle around y-axis")
    parser.add_argument("--theta_z", type=float, default=0, help="Rotation angle around z-axis")
    parser.add_argument("--conductance_parameter", type=float, default=3.0, help="Conductance parameter for anisotropic diffusion")
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="Number of smoothing iterations")
    parser.add_argument("--time_step", type=float, default=0.0625, help="Time step for anisotropic diffusion")
    parser.add_argument("--threshold_filter", type=str, default="Otsu", choices=["Otsu", "Binary"], help="Threshold filter method")
    parser.add_argument("--mip_slices", type=int, default=5, help="Number of slices for maximum intensity projection")
    args = parser.parse_args()
    
    result = main(args.img_path, args.age, args.output_path, args.neonatal,args.months,
            args.ranges,
                  theta_x=args.theta_x, theta_y=args.theta_y, theta_z=args.theta_z,
                  conductance_parameter=args.conductance_parameter,
                  smoothing_iterations=args.smoothing_iterations,
                  time_step=args.time_step,
                  threshold_filter=args.threshold_filter,
                  mip_slices=args.mip_slices
                  )
    
    if result:
        print(f'circumference {result}')
    else:
        print("Failed to process image.")

