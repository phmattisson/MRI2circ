# MRI2circ


# Build docker 
docker build -t mri2circ -f Dockerfile .

# Inputs
T1 image (nifti)
Age in years (float): Template ranges 0-2,3-7,8-13,14-35
--neonatal: optional, if present Age should be entered in weeks as an integer value ranging from 36-44

# Run docker

docker run -v /path/to/input/data:/input -v /path/to/output:/output mri2circ python3.9 /executables/run_process_image_with_options.py --img_path /input/your_image.nii --age 40 --output_path /output --threshold_filter Binary


docker run -v /home/philip-mattisson/Desktop/data:/data mri2circ python3.9 /executables/run_process_image_with_options.py --img_path /data/sub-pixar066_anat_sub-pixar066_T1w.nii.gz --age 35 --output_path /data/V2Out



# References
Automated temporalis muscle quantification and growth charts for children through adulthood doi.org/10.1038/s41467-023-42501-1