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
import subprocess
import matplotlib
matplotlib.use('Agg')
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
from mips import process_and_visualize, ThresholdFilter,length_of_contour_with_spacing,distance_2d_with_spacing,distance_2d_with_spacing

def register_images(fixed_image_path, moving_image_path):
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')
    return registration['warpedmovout']

def select_template_based_on_age(age, neonatal, months):
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
            "../shared_data/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}
        }
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
         threshold_filter=ThresholdFilter.Binary, mip_slices=5):
    
    template_path = select_template_based_on_age(age, neonatal, months)
    registered_image = register_images(template_path, img_path)

    patient_id = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(output_path, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    registered_path = os.path.join(output_dir, 'registered.nii.gz')
    ants.image_write(registered_image, registered_path)

    if not os.path.exists(output_dir + "/no_z"):
        os.mkdir(output_dir + "/no_z")

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

        series_w = np.dstack([funcy(im) for im in series_n])
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

    print(f'Threshold filter: {threshold_filter_enum}')

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

        def find_contour_line_intersections(cx, cy, angle_degs, contour_array, spacing):
            """
            Find intersection points between a line and the brain contour with symmetric angle handling.
            """
            height, width = contour_array.shape
            angle_rads = np.radians(angle_degs)

            # Get contour points
            y_indices, x_indices = np.where(contour_array > 0)
            contour_points = np.column_stack((x_indices, y_indices))  # x, y
            center = np.array([cx, cy])

            vectors = contour_points - center
            point_angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            # Normalize angles to [0, 2π]
            point_angles = np.where(point_angles < 0, point_angles + 2*np.pi, point_angles)
            target_angle = angle_rads if angle_rads >= 0 else angle_rads + 2*np.pi

            # Find points with angles closest to target_angle and target_angle + π
            angle_diff = np.minimum(
                np.abs(point_angles - target_angle),
                np.abs(point_angles - (target_angle + 2*np.pi))
            )
            opposite_diff = np.minimum(
                np.abs(point_angles - (target_angle + np.pi)),
                np.abs(point_angles - (target_angle - np.pi))
            )

            point1_idx = np.argmin(angle_diff)
            point2_idx = np.argmin(opposite_diff)

            x1, y1 = contour_points[point1_idx]
            x2, y2 = contour_points[point2_idx]

            dx_mm = (x2 - x1) * spacing[0]
            dy_mm = (y2 - y1) * spacing[1]
            length_mm = np.sqrt(dx_mm**2 + dy_mm**2)

            return (x1, y1, x2, y2, length_mm)
        
        result_60 = find_contour_line_intersections(center_x_idx, center_y_idx, 60, contour_array, spacing)
        result_120 = find_contour_line_intersections(center_x_idx, center_y_idx, 120, contour_array, spacing)

        plt.ioff()
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        ax1.imshow(mip_array, cmap='gray')
        ax1.contour(contour_array, colors='r')

        ax1.plot([minx_idx, maxx_idx], [center_y_idx, center_y_idx], 'b-', linewidth=2, label=f'Width {width_mm} mm')
        ax1.plot([center_x_idx, center_x_idx], [miny_idx, maxy_idx], 'g-', linewidth=2, label=f'Length {length_mm:} mm')

        if result_60 is not None:
            x1_60, y1_60, x2_60, y2_60, diag_60_mm = result_60
            ax1.plot([x1_60, x2_60], [y1_60, y2_60], 'y--', linewidth=2, 
                    label=f'30° from Vertical (1): {diag_60_mm:.2f} mm')

        if result_120 is not None:
            x1_120, y1_120, x2_120, y2_120, diag_120_mm = result_120
            ax1.plot([x1_120, x2_120], [y1_120, y2_120], 'c--', linewidth=2,
                    label=f'30° from Vertical (2): {diag_120_mm:.2f} mm')
            
        ax1.set_title(f"Head Circumference: {circumference:.3f}")
        ax1.legend()

        # Quadrant plot
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

        ax2.imshow(quadrant_mask, alpha=0.1)
        for qname, qmask in quadrant_masks.items():
            if np.sum(qmask & (largest_component_array > 0)) > 0:
                y_coords, x_coords = np.where(qmask & (largest_component_array > 0))
                x_pos = np.mean(x_coords)
                y_pos = np.mean(y_coords)
                ax2.text(x_pos, y_pos, f'{qname}\n{(np.sum(qmask & (largest_component_array > 0)) * spacing[0] * spacing[1] / 100):.2f} cm²',
                        color='white', fontsize=8, ha='center', va='center')

        ax2.set_title("Head Contour Split into 4 Quadrants")
        ax2.axis('off')

        plt.tight_layout()
        plot_filename = f'combined_visualization_{k}_{os.path.basename(registered_path)}.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close(fig)

        # -------------------------------------------
        # Integration of the image styling steps here
        # -------------------------------------------
        # Convert mip_array to BGR
        mip_normalized = (mip_array / np.max(mip_array) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(mip_normalized, cv2.COLOR_GRAY2BGR)

        # Create alpha channel from contour_array
        alpha = (contour_array > 0).astype(np.uint8) * 255

        # Now apply the transformations as in your snippet
        contours_cv = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_cv = contours_cv[0] if len(contours_cv) == 2 else contours_cv[1]

        if len(contours_cv) > 0:
            big_contour = max(contours_cv, key=cv2.contourArea)

            # Smooth contour
            peri = cv2.arcLength(big_contour, True)
            big_contour = cv2.approxPolyDP(big_contour, 0.001 * peri, True)

            # Draw white filled contour on black background
            contour_img = np.zeros_like(alpha)
            cv2.drawContours(contour_img, [big_contour], 0, 255, -1)

            # Dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
            dilate = cv2.morphologyEx(contour_img, cv2.MORPH_DILATE, kernel)

            # Edge outline
            edge = cv2.Canny(dilate, 0, 200)
            # Thicken edge slightly by blurring
            edge = cv2.GaussianBlur(edge, (0,0), sigmaX=0.3, sigmaY=0.3)

            # White background
            result_styled = np.full_like(bgr, (255,255,255))

            # Invert dilated image and blur for shadow
            dilate_inv = 255 - dilate
            dilate_inv = cv2.GaussianBlur(dilate_inv, (0,0), sigmaX=21, sigmaY=21)
            dilate_inv = cv2.merge([dilate_inv,dilate_inv,dilate_inv])

            # Overlay blurred dilated area (shadow)
            result_styled[dilate_inv>0] = dilate_inv[dilate_inv>0]

            # Overlay dilated white region
            result_styled[dilate==255] = (255,255,255)

            # Overlay BGR where contour
            result_styled[contour_img==255] = bgr[contour_img==255]

            # Overlay edge
            result_styled[edge>0] = (0,0,0)

            final_styled_filename = f'final_styled_{k}_{os.path.basename(registered_path)}.png'
            cv2.imwrite(os.path.join(output_dir, final_styled_filename), result_styled)
        # -------------------------------------------

        longer_diag = 0
        smaller_diag = 0
        if result_60 is not None and result_120 is not None:
            diag_60_mm = result_60[-1]
            diag_120_mm = result_120[-1]
            longer_diag = max(diag_60_mm, diag_120_mm)
            smaller_diag = min(diag_60_mm, diag_120_mm)
        else:
            diag_60_mm = 0
            diag_120_mm = 0

        CVAI = ((longer_diag - smaller_diag)/longer_diag * 100) if longer_diag != 0 else 0

        measurements = {
            'patient_id': patient_id,
            'iteration': k,
            'slice_number': slice_label+k,
            'circumference_mm': circumference,
            'width_mm': width_mm,
            'length_mm': length_mm,
            'diagonal 1 (60 degree)': diag_60_mm,
            'diagonal 2 (120 degree)': diag_120_mm,
            'CVAI': CVAI,
            'area_cm2': area_cm2,
            'cephalic_index': (width_mm / length_mm) * 100 if length_mm != 0 else np.nan,
            'upper_left_area_cm2': area_upper_left,
            'upper_right_area_cm2': area_upper_right,
            'lower_left_area_cm2': area_lower_left,
            'lower_right_area_cm2': area_lower_right,
        }
        all_measurements.append(measurements)

        if slice_label+k == image_array.shape[2]:
            break

    if len(all_measurements) > 0:
        df_results = pd.DataFrame(all_measurements)
        summary_stats = {
            'patient_id': patient_id,
            'circumference_median': df_results['circumference_mm'].median(),
            'circumference_mean': df_results['circumference_mm'].mean(),
            'circumference_std': df_results['circumference_mm'].std(),
            'area_median': df_results['area_cm2'].median(),
            'area_mean': df_results['area_cm2'].mean(),
            'area_std': df_results['area_cm2'].std(),
            **volumes
        }
        df_results.to_csv(os.path.join(output_dir, 'detailed_measurements.csv'), index=False)

        # Save summary statistics
        pd.DataFrame([summary_stats]).to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    return circumference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate brain perimeter from MRI image.")
    parser.add_argument("img_path", type=str, help="Path to the MRI image file")
    parser.add_argument("age", type=int, help="Age of the subject")
    parser.add_argument("output_path", type=str, help="Path to the output folder")
    parser.add_argument("--neonatal", action="store_true", help="Flag to indicate if the subject is neonatal")
    parser.add_argument("--months", action="store_true", help="Flag to indicate if age is specified in months")
    parser.add_argument("--ranges", action="store_true", help='flag to indicate use of ranges instead of slice picker') 
    parser.add_argument("--theta_x", type=float, default=0, help="Rotation angle around x-axis")
    parser.add_argument("--theta_y", type=float, default=0, help="Rotation angle around y-axis")
    parser.add_argument("--theta_z", type=float, default=0, help="Rotation angle around z-axis")
    parser.add_argument("--conductance_parameter", type=float, default=3.0, help="Conductance parameter for anisotropic diffusion")
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="Number of smoothing iterations")
    parser.add_argument("--time_step", type=float, default=0.0625, help="Time step for anisotropic diffusion")
    parser.add_argument("--threshold_filter", type=str, default="Otsu", choices=["Otsu", "Binary"], help="Threshold filter method")
    parser.add_argument("--mip_slices", type=int, default=5, help="Number of slices for maximum intensity projection")
    args = parser.parse_args()


    result = main(args.img_path, args.age, args.output_path, args.neonatal, args.months,
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
