# MRI2circ


# Build docker 
docker build -t mri2circ -f Dockerfile .

# Inputs
T1 image (nifti)
Age in years (float): Template ranges 0-2,3-7,8-13,14-35
--neonatal: optional, if present Age should be entered in weeks as an integer value ranging from 36-44




# Run docker

1. `img_path` (optional, default: "/data/input.nii.gz"): Path to the MRI image file
2. `age` (optional, default: 35): Age of the subject
3. `output_path` (optional, default: "/data/output"): Path to the output folder

Optional named arguments:

- `--neonatal`: Flag to indicate if the subject is neonatal
- `--theta_x THETA_X` (default: 0): Rotation angle around x-axis
- `--theta_y THETA_Y` (default: 0): Rotation angle around y-axis
- `--theta_z THETA_Z` (default: 0): Rotation angle around z-axis
- `--conductance_parameter CONDUCTANCE_PARAMETER` (default: 3.0): Conductance parameter for anisotropic diffusion
- `--smoothing_iterations SMOOTHING_ITERATIONS` (default: 5): Number of smoothing iterations
- `--time_step TIME_STEP` (default: 0.0625): Time step for anisotropic diffusion
- `--threshold_filter {Otsu,Binary}` (default: "Otsu"): Threshold filter method
- `--mip_slices MIP_SLICES` (default: 5): Number of slices for maximum intensity projection


minimal docker run:
docker run -v /home/philip-mattisson/Desktop/data:/data \
           mri2circ \
           python3.9 /executables/new_process_image_with_options.py \
           /data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz 35 /data/V2Out

all arguments

docker run -v /home/philip-mattisson/Desktop/data:/data \
           mri2circ \
           python3.9 /executables/new_process_image_with_options.py \
           /data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz \
           35 \
           /data/V2Out \
           --neonatal \
           --theta_x 10 \
           --theta_y 5 \
           --theta_z 15 \
           --conductance_parameter 2.5 \
           --smoothing_iterations 7 \
           --time_step 0.05 \
           --threshold_filter Otsu \
           --mip_slices 3






# References
Automated temporalis muscle quantification and growth charts for children through adulthood doi.org/10.1038/s41467-023-42501-1