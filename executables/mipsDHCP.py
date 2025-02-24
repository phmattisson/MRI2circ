import argparse
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev

class ThresholdFilter(Enum):
    Otsu = 1
    Binary = 2

def process_and_visualize(image_path, slice_num, theta_x=0, theta_y=0, theta_z=0,
                          conductance_parameter=3.0, smoothing_iterations=5, time_step=0.0625,
                          threshold_filter=ThresholdFilter.Otsu, mip_slices=5):
    # Load the image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    
    # Apply rotation
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetRotation(np.radians(theta_x), np.radians(theta_y), np.radians(theta_z))
    euler_transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint(
        [(dim - 1) / 2.0 for dim in image.GetSize()]
    ))
    rotated_image = sitk.Resample(image, euler_transform)
    rotated_array = sitk.GetArrayFromImage(rotated_image)

    # Perform MIP
    if slice_num == -1:
        slice_num = rotated_array.shape[0] // 2
    start_slice = max(0, slice_num - mip_slices // 2)
    end_slice = min(rotated_array.shape[0], slice_num + mip_slices // 2 + 1)
    mip_array = np.max(rotated_array[start_slice:end_slice], axis=0)

    # Normalize intensity to 0-255 range
    mip_normalized = cv2.normalize(mip_array, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20,20))
    mip_enhanced = clahe.apply(mip_normalized.astype(np.uint8))
    
    # Convert back to SimpleITK image
    mip_slice = sitk.GetImageFromArray(mip_enhanced)

    # Apply edge-preserving smoothing with adaptive conductance
    smooth_slice = sitk.CurvatureAnisotropicDiffusion(
        sitk.Cast(mip_slice, sitk.sitkFloat64),
        conductanceParameter=conductance_parameter,
        numberOfIterations=smoothing_iterations,
        timeStep=time_step
    )

    # Apply multi-level Otsu thresholding
    if threshold_filter == ThresholdFilter.Otsu:
        # Use 3-level Otsu and take the first threshold
        otsu_filter = sitk.OtsuMultipleThresholdsImageFilter()
        otsu_filter.SetNumberOfThresholds(2)
        multi_thresholded = otsu_filter.Execute(smooth_slice)
        thresholded = multi_thresholded > 0
    else:
        # Adaptive binary thresholding
        smooth_array = sitk.GetArrayFromImage(smooth_slice)
        binary = cv2.adaptiveThreshold(
            smooth_array.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,  # block size
            -2   # constant subtracted from mean
        )
        thresholded = sitk.GetImageFromArray(binary)

    # Rest of the processing remains the same...
    hole_filling = sitk.BinaryGrindPeak(thresholded)

    if sitk.GetArrayFromImage(hole_filling)[0,0] == 1:
        hole_filling = sitk.Not(hole_filling)

    component_image = sitk.ConnectedComponent(hole_filling)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component = sorted_component_image == 1
    largest_component_array = sitk.GetArrayFromImage(largest_component)

    contour_image = sitk.BinaryContour(largest_component)
    contour_array = sitk.GetArrayFromImage(contour_image)

    spacing = image.GetSpacing()
    
    # More aggressive morphological operations for smoother contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    smoothed_mask = cv2.morphologyEx(largest_component_array.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(smoothed_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    if len(contours) == 0:
        return 0, contour_array, mip_array, spacing, largest_component_array

    # More aggressive smoothing for contours
    smoothened = []
    for contour in contours:
        x, y = contour[:,0,0], contour[:,0,1]
        x = x.tolist()
        y = y.tolist()
        if len(x) > 4:
            tck, u = splprep([x, y], u=None, s=100.0, per=0)  # Increased smoothing
            u_new = np.linspace(u.min(), u.max(), 200)
            x_new, y_new = splev(u_new, tck, der=0)
            res_array = np.array([[[int(ix), int(iy)]] for ix, iy in zip(x_new, y_new)], dtype=np.int32)
            smoothened.append(res_array)
        else:
            smoothened.append(contour)

    circumference = length_of_contour_with_spacing(smoothened, spacing[0], spacing[1])
    return circumference, contour_array, mip_enhanced, spacing, largest_component_array, smoothened

def length_of_contour_with_spacing(contours, x_spacing, y_spacing):
    # Assuming only one main contour or taking the largest
    if len(contours) == 0:
        return 0
    # Choose the largest contour by area
    largest = max(contours, key=cv2.contourArea)

    arc_length = 0
    for i in range(len(largest)):
        p1 = largest[i][0]
        p2 = largest[(i+1) % len(largest)][0]
        arc_length += distance_2d_with_spacing(p1, p2, x_spacing, y_spacing)
    return arc_length

def distance_2d_with_spacing(p1, p2, x_spacing, y_spacing):
    return np.sqrt(
        (x_spacing * (p1[0] - p2[0])) ** 2 + (y_spacing * (p1[1] - p2[1])) ** 2
    )



if __name__ == "__main__":
    circumference, contour, mip, spacing, largest_component_array,smoothed = process_and_visualize(
        #'/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/T1.nii/registered.nii.gz',
        #"/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/sub-pixar066_anat_sub-pixar066_T1w.nii/registered.nii.gz",
        "/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/outlierDHCP/registered.nii.gz",
        slice_num=70,
        theta_x=0,
        theta_y=0,
        theta_z=0,
        conductance_parameter=3.0,
        smoothing_iterations=5,
        time_step=0.0625,
        threshold_filter=ThresholdFilter.Otsu,
        mip_slices=5
    )

    print(f"Calculated Head Circumference: {circumference:.2f} mm")

    # Show result
    """
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Smoothed Contour Overlay')
    plt.show()
    """
    plt.figure(figsize=(12, 4))

    # Plot 1: Original MIP
    plt.subplot(131)
    plt.imshow(mip, cmap='gray')
    plt.title('Maximum Intensity Projection')
    plt.axis('off')

    # Plot 2: Contour overlay
    plt.subplot(132)
    plt.imshow(mip, cmap='gray')
    plt.imshow(contour, cmap='hot', alpha=0.3)
    plt.title(f'Contour (Circumference: {circumference:.2f} mm)')
    plt.axis('off')

    # Plot 3: Binary mask with contour
    plt.subplot(133)
    plt.imshow(largest_component_array, cmap='gray')
    plt.imshow(contour, cmap='hot', alpha=0.3)
    plt.title('Binary Mask with Contour')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


