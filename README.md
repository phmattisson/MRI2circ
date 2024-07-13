# MRI2circ


# Build docker 
docker build -t mri2circ -f Dockerfile .

# Inputs
T1 image (nifti)
Age in years (float): Template ranges 0-2,3-7,8-13,14-35

# Run docker

docker run \
  -v /path/to/input/directory:/input \
  -v /path/to/output/directory:/output \
  mri2circ \
  python3.9 /executables/process_image.py \
  /input/input_file.nii.gz age /output


# References
Automated temporalis muscle quantification and growth charts for children through adulthood doi.org/10.1038/s41467-023-42501-1