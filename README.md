# MRI2circ


# Build docker 
docker build -t mri2circ -f Dockerfile .

# Inputs
T1 image (nifti)
Age (float)

# Run docker

docker run \
  -v /path/to/input/directory:/input \
  -v /path/to/output/directory:/output \
  mri2circ \
  python3.9 /executables/process_image.py \
  /input/input_file.nii.gz age /output