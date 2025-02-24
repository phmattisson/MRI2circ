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

    # Invert if necessary
    if sitk.GetArrayFromImage(hole_filling)[0,0] == 1:
        hole_filling = sitk.Not(hole_filling)

    # Select largest component
    component_image = sitk.ConnectedComponent(hole_filling)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component = sorted_component_image == 1
    largest_component_array = sitk.GetArrayFromImage(largest_component)

    # Get contour (binary contour)
    contour_image = sitk.BinaryContour(largest_component)
    contour_array = sitk.GetArrayFromImage(contour_image)


    # Get original spacing
    spacing = image.GetSpacing()
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    #smoothed_mask = cv2.morphologyEx(largest_component_array.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    smoothed_mask = cv2.morphologyEx(largest_component_array.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    #smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
   
    # Convert contour_array to OpenCV contours
    contours, hierarchy = cv2.findContours(smoothed_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # If no contours found, return early
    if len(contours) == 0:
        return 0, contour_array, mip_array, spacing, largest_component_array

    # Smooth the contour using splprep/splev (from second snippet)
    smoothened = []
    for contour in contours:
        x, y = contour[:,0,0], contour[:,0,1]
        # Convert to lists
        x = x.tolist()
        y = y.tolist()
        # Perform spline smoothing
        if len(x) > 4:  # Need a few points for spline
            tck, u = splprep([x, y], u=None,k=1,s=20, per=1)
            u_new = np.linspace(u.min(), u.max(), 40)  # more points for a smoother contour
            x_new, y_new = splev(u_new, tck, der=0)
            # Convert back to integer points
            res_array = np.array([[[int(ix), int(iy)]] for ix, iy in zip(x_new, y_new)], dtype=np.int32)
            smoothened.append(res_array)
        else:
            # If too few points, just keep original
            smoothened.append(contour)

    # Replace original contours with smoothed
    contours = smoothened

    # Now we have a "result" image with smoothed contour and stylized edges

    # Calculate circumference of the smoothed contour
    circumference = length_of_contour_with_spacing(contours, spacing[0], spacing[1])

    # Return circumference, contour_array, mip_array, spacing, largest_component_array
    return circumference, contour_array, mip_array, spacing, largest_component_array,smoothened

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

def plot_circumference_on_mip(mip_array, smoothened, circumference):
    # Normalize and convert the MIP array to 8-bit for display
    norm_mip = cv2.normalize(mip_array, None, 0, 255, cv2.NORM_MINMAX)
    norm_mip = norm_mip.astype(np.uint8)

    # Convert grayscale MIP to BGR so we can draw colored contours
    mip_bgr = cv2.cvtColor(norm_mip, cv2.COLOR_GRAY2BGR)

    # Draw the smoothed contour on the image (e.g., green color, thickness=2)
    cv2.drawContours(mip_bgr, smoothened, -1, (0,255,0), 2)

    # Convert BGR to RGB for matplotlib display
    mip_rgb = cv2.cvtColor(mip_bgr, cv2.COLOR_BGR2RGB)

    # Display the image with the contour and print the circumference
    plt.figure(figsize=(10,10))
    plt.imshow(mip_rgb)
    plt.title(f"Circumference: {circumference:.2f} mm")
    plt.axis('off')  # Hide axis if desired
    plt.show()



if __name__ == "__main__":
    circumference, contour, mip, spacing,largest_component_array,smoothed = process_and_visualize(
        #'/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/T1.nii/registered.nii.gz',
        #"/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/sub-pixar066_anat_sub-pixar066_T1w.nii/registered.nii.gz",
        #"/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/outlierDHCP/registered.nii.gz",
        #"/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/UKB1009121_T1.nii/registered.nii.gz",
        "/Users/philipmattisson/offline/OfflineCentile/mri2circV2/MRI2circ/output/UKB1010001_T1.nii/registered.nii.gz",
        slice_num=65,
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


    

    plot_circumference_on_mip(mip, smoothed, circumference)


