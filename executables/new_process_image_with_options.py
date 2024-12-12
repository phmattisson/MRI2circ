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
from mips import process_circumference, ThresholdFilter,length_of_contour_with_spacing,distance_2d_with_spacing,distance_2d_with_spacing

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

    
    circumference,contour_array, mip_array,spacing,largest_component_array = process_circumference(
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
    """
    y_indices, x_indices = np.where(contour_array > 0)

    # Get the spacing from the original image
    spacing_x = spacing[0]  # Spacing along the x-axis (width)
    spacing_y = spacing[1]  # Spacing along the y-axis (height)

    # Convert pixel indices to physical coordinates using spacing
    coordinates = [(x * spacing_x, y * spacing_y) for x, y in zip(x_indices, y_indices)]

    # Create a polygon from the coordinates
    polygon = Polygon(coordinates)

    minx, miny, maxx, maxy = polygon.bounds

    # Calculate width and length of the bounding box in millimeters
    width_mm = maxx - minx
    length_mm = maxy - miny
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    vertical_line = LineString([(center_x, miny), (center_x, maxy)])
    horizontal_line = LineString([(minx, center_y), (maxx, center_y)])

    # Split the polygon with the vertical line
    split_polygons = split(polygon, vertical_line)


    # Optionally, calculate the perimeter (circumference) using Shapely
    area_pixels = np.sum(largest_component_array)
    # Convert area to square millimeters
    area_mm2 = (area_pixels * spacing_x * spacing_y)/100
    cephalic_index = (width_mm/length_mm)*100

    # Save results including width and length
    df_results = pd.DataFrame({
        'patient_id': [patient_id],
        'circumference (mm)': [circumference],  # You can replace this with 'perimeter_mm' if you prefer Shapely's calculation
        'width (mm)': [width_mm],
        'length (mm)': [length_mm],
        'area (cm^2)': [area_mm2],
        'cephalic_index': [cephalic_index]
    })
    df_results.to_csv(os.path.join(output_dir, 'circumference.csv'), index=False)

    # Plot the visualization with width and length measurements
    plt.ioff()
    plt.figure(figsize=(10, 10))
    plt.imshow(mip_array, cmap='gray')
    plt.contour(contour_array, colors='r')

    # Convert physical coordinates back to pixel indices for plotting
    minx_idx = minx / spacing_x
    maxx_idx = maxx / spacing_x
    miny_idx = miny / spacing_y
    maxy_idx = maxy / spacing_y

    # Plot width line (horizontal line at the middle y-coordinate)
    mid_y_idx = (miny_idx + maxy_idx) / 2
    plt.plot([minx_idx, maxx_idx], [mid_y_idx, mid_y_idx], 'b-', linewidth=2)

    # Plot length line (vertical line at the middle x-coordinate)
    mid_x_idx = (minx_idx + maxx_idx) / 2
    plt.plot([mid_x_idx, mid_x_idx], [miny_idx, maxy_idx], 'g-', linewidth=2)

    # Add title with measurements
    plt.title(f"Head Circumference: {circumference:.2f} mm\nWidth: {width_mm:.2f} mm, Length: {length_mm:.2f} mm")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'contour_visualization.png'))
    plt.close()

    print(f"Circumference: {circumference:.2f} mm")
    print(f"Width: {width_mm:.2f} mm")
    print(f"Length: {length_mm:.2f} mm")
    print(f'Area cm^2: {area_mm2}')
    
    """
   # Get the spacing from the original image
    spacing_x = spacing[0]  # Spacing along the x-axis (width)
    spacing_y = spacing[1]  # Spacing along the y-axis (height)

    # Calculate the indices corresponding to the head contour
    y_indices, x_indices = np.where(contour_array > 0)

    # Calculate the bounding box of the head contour
    minx_idx = np.min(x_indices)
    maxx_idx = np.max(x_indices)
    miny_idx = np.min(y_indices)
    maxy_idx = np.max(y_indices)

    # Calculate width and length in millimeters
    width_mm = (maxx_idx - minx_idx) * spacing_x
    length_mm = (maxy_idx - miny_idx) * spacing_y
    center_x_idx = (minx_idx + maxx_idx) // 2
    center_y_idx = (miny_idx + maxy_idx) // 2

    # Calculate total area in pixels
    area_pixels = np.sum(largest_component_array)
    # Convert area to square millimeters
    area_mm2 = area_pixels * spacing_x * spacing_y
    area_cm2 = area_mm2 / 100

    # Create masks for each quadrant
    print(type(largest_component_array))
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

    # Calculate area in pixels for each quadrant
    area_upper_left_pixels = np.sum(largest_component_array & upper_left_mask)
    area_upper_right_pixels = np.sum(largest_component_array & upper_right_mask)
    area_lower_left_pixels = np.sum(largest_component_array & lower_left_mask)
    area_lower_right_pixels = np.sum(largest_component_array & lower_right_mask)

    # Convert area to cm^2
    area_upper_left_cm2 = area_upper_left_pixels * spacing_x * spacing_y / 100
    area_upper_right_cm2 = area_upper_right_pixels * spacing_x * spacing_y / 100
    area_lower_left_cm2 = area_lower_left_pixels * spacing_x * spacing_y / 100
    area_lower_right_cm2 = area_lower_right_pixels * spacing_x * spacing_y / 100

    # Store the areas in a dictionary
    quadrant_areas = {
        'upper_left': area_upper_left_cm2,
        'upper_right': area_upper_right_cm2,
        'lower_left': area_lower_left_cm2,
        'lower_right': area_lower_right_cm2
    }

    # Save results including width, length, and quadrant areas
    df_results = pd.DataFrame({
        'patient_id': [patient_id],
        'circumference (mm)': [circumference],
        'width (mm)': [width_mm],
        'length (mm)': [length_mm],
        'area (cm^2)': [area_cm2],
        'cephalic_index': [(width_mm / length_mm) * 100],
        'upper_left_area (cm^2)': [quadrant_areas.get('upper_left', 0)],
        'upper_right_area (cm^2)': [quadrant_areas.get('upper_right', 0)],
        'lower_left_area (cm^2)': [quadrant_areas.get('lower_left', 0)],
        'lower_right_area (cm^2)': [quadrant_areas.get('lower_right', 0)]
    })
    df_results.to_csv(os.path.join(output_dir, 'circumference_and_quadrant_areas.csv'), index=False)

    # Plot the visualization with width and length measurements
    plt.ioff()
    plt.figure(figsize=(10, 10))
    plt.imshow(mip_array, cmap='gray')
    plt.contour(contour_array, colors='r')

    # Plot width line (horizontal line at the middle y-coordinate)
    plt.plot([minx_idx, maxx_idx], [center_y_idx, center_y_idx], 'b-', linewidth=2)

    # Plot length line (vertical line at the middle x-coordinate)
    plt.plot([center_x_idx, center_x_idx], [miny_idx, maxy_idx], 'g-', linewidth=2)

    # Plot the vertical and horizontal lines for quadrants
    plt.plot([center_x_idx, center_x_idx], [0, mip_array.shape[0]], 'y--', linewidth=1)  # Vertical line
    plt.plot([0, mip_array.shape[1]], [center_y_idx, center_y_idx], 'y--', linewidth=1)  # Horizontal line

    # Annotate quadrants
    quadrant_positions = {
        'upper_left': (minx_idx + (center_x_idx - minx_idx) / 2, miny_idx + (center_y_idx - miny_idx) / 2),
        'upper_right': (center_x_idx + (maxx_idx - center_x_idx) / 2, miny_idx + (center_y_idx - miny_idx) / 2),
        'lower_left': (minx_idx + (center_x_idx - minx_idx) / 2, center_y_idx + (maxy_idx - center_y_idx) / 2),
        'lower_right': (center_x_idx + (maxx_idx - center_x_idx) / 2, center_y_idx + (maxy_idx - center_y_idx) / 2),
    }

    for label, (x_pos, y_pos) in quadrant_positions.items():
        area = quadrant_areas.get(label, 0)
        plt.text(x_pos, y_pos, f'{label.replace("_", " ")}\n{area:.2f} cm²', color='white', fontsize=8, ha='center', va='center')

    # Add title with measurements
    plt.title(f"Head Circumference: {circumference:.2f} mm\nWidth: {width_mm:.2f} mm, Length: {length_mm:.2f} mm")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'contour_and_quadrants_visualization.png'))
    plt.close()

    # Plot the quadrants with different colors
    plt.figure(figsize=(10, 10))
    plt.imshow(mip_array, cmap='gray')
    plt.contour(contour_array, colors='r')

    # Create an RGB image to overlay quadrants
 

    # Create a colored mask for visualization
    colored_mask = np.zeros((largest_component_array.shape[0], largest_component_array.shape[1], 3), dtype=np.uint8)

    colors = {
        'upper_left': [255, 0, 0],    # Red
        'upper_right': [0, 255, 0],   # Green
        'lower_left': [0, 0, 255],    # Blue
        'lower_right': [255, 255, 0]  # Yellow
    }

    # Apply colors to the quadrants
    for label, color in colors.items():
        if label == 'upper_left':
            mask = largest_component_array & upper_left_mask
        elif label == 'upper_right':
            mask = largest_component_array & upper_right_mask
        elif label == 'lower_left':
            mask = largest_component_array & lower_left_mask
        elif label == 'lower_right':
            mask = largest_component_array & lower_right_mask

        colored_mask[mask > 0] = color

    # Overlay the colored mask onto the image
    plt.imshow(colored_mask, alpha=0.5)

    # Annotate quadrants with areas
    for label, (x_pos, y_pos) in quadrant_positions.items():
        area = quadrant_areas.get(label, 0)
        plt.text(x_pos, y_pos, f'{label.replace("_", " ")}\n{area:.2f} cm²', color='white', fontsize=8, ha='center', va='center')

    plt.title("Head Contour Split into Quadrants")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'quadrant_visualization.png'))
    plt.close()

    # Print out the measurements
    print(f"Circumference: {circumference:.2f} mm")
    print(f"Width: {width_mm:.2f} mm")
    print(f"Length: {length_mm:.2f} mm")
    print(f"Area (cm^2): {area_cm2:.2f}")
    print(f"Cephalic Index: {(width_mm / length_mm) * 100:.2f}")
    for label, area in quadrant_areas.items():
        print(f"Area of {label.replace('_', ' ')}: {area:.2f} cm^2")

    return circumference

# Get the results dictionary from process_and_visualize
"""     results = process_and_visualize(
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

    # Extract values from the dictionary
    circumference = results['circumference']
    width = results['width']
    length = results['length']
    area = results['area']
    contour_array = results['contour_array']
    mip_array = results['mip_array']
    polygon = results['polygon']
    bounds = results['bounds']

    print(f"width: {width}")
    print(f"length: {length}")
    print(f"area: {area}")

    # Save results
    df_results = pd.DataFrame({
        'patient_id': [patient_id],
        'circumference': [circumference],
        'width': [width],
        'length': [length],
        'area': [area]
    })
    df_results.to_csv(os.path.join(output_dir, 'measurements.csv'), index=False)

    # Save the visualization
    plt.ioff()
    plt.figure(figsize=(10, 10))
    plt.imshow(mip_array, cmap='gray')
    plt.contour(contour_array, colors='r')
    plt.title(f"Head Measurements:\nCircumference: {circumference:.2f} mm\n"
            f"Width: {width:.2f} mm\nLength: {length:.2f} mm\n"
            f"Area: {area:.2f} mm²")
    plt.axis('off')
    plt.savefig   """  
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

