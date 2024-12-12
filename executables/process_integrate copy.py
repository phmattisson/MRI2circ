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
from shapely.geometry import Polygon,LineString,MultiPolygon,GeometryCollection
from shapely.ops import split
from shapely.affinity import scale
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
# Import necessary functions from mips.py
from mips import process_and_visualize, ThresholdFilter,length_of_contour_with_spacing,distance_2d_with_spacing,distance_2d_with_spacing

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

def main(img_path, age, output_path, neonatal,months,ranges,theta_x=0, theta_y=0, theta_z=0, 
         conductance_parameter=3.0, smoothing_iterations=5, time_step=0.0625, 
         threshold_filter=ThresholdFilter.Otsu, mip_slices=5):
    # Select template
    template_path = select_template_based_on_age(age, neonatal,months)

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
    print(slice_label)
      
        # Calculate circumference using mips.py function (on non-enhanced registered image)
    if args.threshold_filter == "Otsu":
        threshold_filter_enum = ThresholdFilter.Otsu
    elif args.threshold_filter == "Binary":
        threshold_filter_enum = ThresholdFilter.Binary
    else:
        raise ValueError(f"Invalid threshold filter: {args.threshold_filter}")

    if ranges:
        slice_interval = list(range(88,108))
        results = []
        for slice_range in slice_interval:
            circumference, contour_array, mip_array,spacing,largest_component_array = process_and_visualize(
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


    all_measurements = []

    circumference = 1000
    k = 0
    volumes = {
        'posterior_left': 0,
        'posterior_right': 0,
        'anterior_left': 0,
        'anterior_right': 0,
        'total': 0
    }

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
        except:
            print('Error with circumference calculation. Saving results...')
            if len(all_measurements) > 0:
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

        # Calculate basic measurements
        y_indices, x_indices = np.where(contour_array > 0)

        # Calculate bounding box
        minx_idx = np.min(x_indices)
        maxx_idx = np.max(x_indices)
        miny_idx = np.min(y_indices)
        maxy_idx = np.max(y_indices)

        height, width = largest_component_array.shape

        # Dimensions in mm
        width_mm = (maxx_idx - minx_idx) * spacing[0]
        length_mm = (maxy_idx - miny_idx) * spacing[1]

        center_x_idx = (minx_idx + maxx_idx) // 2
        center_y_idx = (miny_idx + maxy_idx) // 2

        # Define quadrant masks
        Y, X = np.ogrid[:height, :width]
        upper_mask = Y < center_y_idx
        lower_mask = Y >= center_y_idx
        left_mask = X < center_x_idx
        right_mask = X >= center_x_idx

        upper_left_mask = upper_mask & left_mask
        upper_right_mask = upper_mask & right_mask
        lower_left_mask = lower_mask & left_mask
        lower_right_mask = lower_mask & right_mask

        # Quadrant areas in cm²
        area_upper_left = np.sum(largest_component_array & upper_left_mask) * spacing[0] * spacing[1] / 100
        area_upper_right = np.sum(largest_component_array & upper_right_mask) * spacing[0] * spacing[1] / 100
        area_lower_left = np.sum(largest_component_array & lower_left_mask) * spacing[0] * spacing[1] / 100
        area_lower_right = np.sum(largest_component_array & lower_right_mask) * spacing[0] * spacing[1] / 100
        area_pixels = np.sum(largest_component_array)

        z_spacing_cm = spacing[2] / 10
        area_mm2 = area_pixels * spacing[0] * spacing[1]
        area_cm2 = area_mm2 / 100

        # Quadrant area dictionary
        quadrant_areas = {
            'posterior_left': area_upper_left,
            'posterior_right': area_upper_right,
            'anterior_left': area_lower_left,
            'anterior_right': area_lower_right
        }

        # Volume contribution for this slice
        slice_contribution = {
            'posterior_left': area_upper_left * z_spacing_cm,
            'posterior_right': area_upper_right * z_spacing_cm,
            'anterior_left': area_lower_left * z_spacing_cm,
            'anterior_right': area_lower_right * z_spacing_cm,
            'total': area_cm2 * z_spacing_cm
        }

        for key in volumes:
            volumes[key] += slice_contribution.get(key, 0)

        # Function to compute intersection points for a given angle
        def compute_intersections(angle_degs, cx, cy, minx, maxx, miny, maxy):
            angle_rads = np.radians(angle_degs)
            tan_angle = np.tan(angle_rads)
            points = []

            # Check intersection with x = minx
            y_int_minx = cy + (minx - cx)*tan_angle
            if miny <= y_int_minx <= maxy:
                points.append((minx, y_int_minx))

            # Check intersection with x = maxx
            y_int_maxx = cy + (maxx - cx)*tan_angle
            if miny <= y_int_maxx <= maxy:
                points.append((maxx, y_int_maxx))

            # If we don't have two points, check horizontal edges
            if len(points) < 2:
                # Solve for x when y = miny
                if abs(tan_angle) > 1e-10:
                    x_int_miny = cx + (miny - cy)/tan_angle
                    if minx <= x_int_miny <= maxx:
                        points.append((x_int_miny, miny))

                    # Solve for x when y = maxy
                    x_int_maxy = cx + (maxy - cy)/tan_angle
                    if minx <= x_int_maxy <= maxx:
                        points.append((x_int_maxy, maxy))

            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                dx_mm = (x2 - x1)*spacing[0]
                dy_mm = (y2 - y1)*spacing[1]
                diag_length_mm = np.sqrt(dx_mm**2 + dy_mm**2)
                return points, diag_length_mm
            else:
                return None, None

        # Compute both diagonals (30° from vertical):
        # Vertical is at 90°, so diagonals are at 90°±30° = 60° and 120° from horizontal
        def find_line_within_mask(cx, cy, angle_degs, largest_component_array, spacing, step=1.0):
            """
            Find the segment of a line inside the largest_component_array mask.
            The line passes through (cx, cy) at angle_degs (from horizontal, in degrees).
            Returns:
                (x1, y1, x2, y2, length_mm)
            or None if no valid segment found.
            """
            angle_rads = np.radians(angle_degs)
            cos_a = np.cos(angle_rads)
            sin_a = np.sin(angle_rads)
            
            height, width = largest_component_array.shape
            
            def in_bounds(x, y):
                return 0 <= x < width and 0 <= y < height
            
            # Search in positive direction
            t = 0
            x_in, y_in = cx, cy
            while True:
                x_test = int(round(cx + t*cos_a))
                y_test = int(round(cy + t*sin_a))
                if not in_bounds(x_test, y_test) or largest_component_array[y_test, x_test] == 0:
                    # We stepped outside the mask. The endpoint is one step before.
                    if t == 0:
                        # The center is already outside or just at the boundary, no valid line
                        pos_end = None
                    else:
                        # Backtrack one step to stay inside
                        t -= step
                        pos_end = (cx + t*cos_a, cy + t*sin_a)
                    break
                t += step
            
            # Search in negative direction
            t = 0
            while True:
                x_test = int(round(cx - t*cos_a))
                y_test = int(round(cy - t*sin_a))
                if not in_bounds(x_test, y_test) or largest_component_array[y_test, x_test] == 0:
                    if t == 0:
                        neg_end = None
                    else:
                        t -= step
                        neg_end = (cx - t*cos_a, cy - t*sin_a)
                    break
                t += step
            
            if pos_end is None or neg_end is None:
                return None
            
            (x1, y1), (x2, y2) = neg_end, pos_end
            
            # Convert pixel distances to mm
            dx_mm = (x2 - x1)*spacing[0]
            dy_mm = (y2 - y1)*spacing[1]
            length_mm = np.sqrt(dx_mm**2 + dy_mm**2)
            return (x1, y1, x2, y2, length_mm)


    # After you have center_x_idx, center_y_idx, and largest_component_array:
    result_60 = find_line_within_mask(center_x_idx, center_y_idx, 60, largest_component_array, spacing)
    result_120 = find_line_within_mask(center_x_idx, center_y_idx, 120, largest_component_array, spacing)

    plt.ioff()
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.imshow(mip_array, cmap='gray')
    ax1.contour(contour_array, colors='r')

    # Plot width and length lines as before
    ax1.plot([minx_idx, maxx_idx], [center_y_idx, center_y_idx], 'b-', linewidth=2, label='Width')
    ax1.plot([center_x_idx, center_x_idx], [miny_idx, maxy_idx], 'g-', linewidth=2, label='Length')

    # Plot first diagonal if found
    if result_60 is not None:
        x1_60, y1_60, x2_60, y2_60, diag_60_mm = result_60
        ax1.plot([x1_60, x2_60], [y1_60, y2_60], 'y--', linewidth=2, label=f'30° from Vertical (1): {diag_60_mm:.2f} mm')

    # Plot second diagonal if found
    if result_120 is not None:
        x1_120, y1_120, x2_120, y2_120, diag_120_mm = result_120
        ax1.plot([x1_120, x2_120], [y1_120, y2_120], 'c--', linewidth=2, label=f'30° from Vertical (2): {diag_120_mm:.2f} mm')

    ax1.set_title(f"Circumference: {circumference:.2f} mm\nWidth: {width_mm:.2f} mm, Length: {length_mm:.2f} mm")
    ax1.axis('off')
    ax1.legend()

    # Quadrant plot remains the same
    ax2.imshow(mip_array, cmap='gray')
    ax2.contour(contour_array, colors='r')

    quadrant_mask = np.zeros((height, width, 3), dtype=np.uint8)
    quadrant_colors = {
        'posterior_left': [255, 0, 0],
        'posterior_right': [0, 255, 0],
        'anterior_left': [0, 0, 255],
        'anterior_right': [255, 255, 0]
    }
    quadrant_masks = {
        'posterior_left': upper_left_mask,
        'posterior_right': upper_right_mask,
        'anterior_left': lower_left_mask,
        'anterior_right': lower_right_mask
    }

    for qname, qmask in quadrant_masks.items():
        quadrant_mask[qmask & (largest_component_array > 0)] = quadrant_colors[qname]

    ax2.imshow(quadrant_mask, alpha=0.4)
    for qname, qmask in quadrant_masks.items():
        if np.sum(qmask & (largest_component_array > 0)) > 0:
            y_coords, x_coords = np.where(qmask & (largest_component_array > 0))
            x_pos = np.mean(x_coords)
            y_pos = np.mean(y_coords)
            ax2.text(x_pos, y_pos, f'{qname}\n{quadrant_areas[qname]:.2f} cm²',
                    color='white', fontsize=8, ha='center', va='center')

    ax2.set_title("Head Contour Split into 4 Quadrants")
    ax2.axis('off')

    plt.tight_layout()
    plot_filename = f'combined_visualization_{k}_{os.path.basename(registered_path)}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close(fig)

    # Once done with the loop, save final results
    if len(all_measurements) > 0:
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
    parser.add_argument("--threshold_filter", type=str, default="Binary", choices=["Otsu", "Binary"], help="Threshold filter method")
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

