import argparse
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import unary_union

class ThresholdFilter(Enum):
    Otsu = 1
    Binary = 2

def process_and_visualize(image_path, slice_num, theta_x=0, theta_y=0, theta_z=0,
                          conductance_parameter=3.0, smoothing_iterations=5, time_step=0.0625,
                          threshold_filter=ThresholdFilter.Otsu, mip_slices=5):
    # Load the image
    #print(f"Image path {image_path}")
    #print(f'conductance parameter {conductance_parameter}')
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    #print(image)

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

    # Convert MIP back to SimpleITK image
    mip_slice = sitk.GetImageFromArray(mip_array)

    # Apply smoothing with conductance parameter
    smooth_slice = sitk.CurvatureAnisotropicDiffusion(
        sitk.Cast(mip_slice, sitk.sitkFloat64),
        conductanceParameter=conductance_parameter,
        numberOfIterations=smoothing_iterations,
        timeStep=time_step
    )

    # Apply thresholding
    if threshold_filter == ThresholdFilter.Otsu:
        thresholded = sitk.OtsuThreshold(smooth_slice)
    else:
        thresholded = sitk.BinaryThreshold(smooth_slice, lowerThreshold=0, upperThreshold=40)

    # Hole filling
    hole_filling = sitk.BinaryGrindPeak(thresholded)

    # Invert the image if necessary
    if sitk.GetArrayFromImage(hole_filling)[0, 0] == 1:
        hole_filling = sitk.Not(hole_filling)

    # Select largest component
    component_image = sitk.ConnectedComponent(hole_filling)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component = sorted_component_image == 1
    largest_component_array = sitk.GetArrayFromImage(largest_component)

    # Get contour
    contour_image = sitk.BinaryContour(largest_component)

    # Convert back to numpy array
    contour_array = sitk.GetArrayFromImage(contour_image)

    # Calculate circumference
    spacing = image.GetSpacing()
    circumference = length_of_contour_with_spacing(contour_array, spacing[0], spacing[1])

    return circumference, contour_array, mip_array,spacing,largest_component_array


def length_of_contour_with_spacing(binary_contour_slice, x_spacing, y_spacing):
    contours, _ = cv2.findContours(binary_contour_slice.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    if len(contours) == 0:
        return 0

    parent_contour = contours[0]
    arc_length = 0

    for i in range(len(parent_contour) - 1):
        arc_length += distance_2d_with_spacing(
            parent_contour[i][0], parent_contour[i + 1][0], x_spacing, y_spacing
        )
    arc_length += distance_2d_with_spacing(
        parent_contour[-1][0], parent_contour[0][0], x_spacing, y_spacing
    )
    return arc_length

def distance_2d_with_spacing(p1, p2, x_spacing, y_spacing):
    return np.sqrt(
        (x_spacing * (p1[0] - p2[0])) ** 2 + (y_spacing * (p1[1] - p2[1])) ** 2
    )

'''
if __name__ == "__main__":
    # Example usage
    circumference, contour, mip = process_and_visualize(
        '/Users/philipmattisson/offline/OfflineCentile/mri2circV2/test_data/testoutput/sub-CC00143BN12_ses-47600_T1w.nii/registered.nii.gz',
        slice_num=149,
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
'''