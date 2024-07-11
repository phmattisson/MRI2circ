CUDA_VISIBLE_DEVICES = "0"

AGE_FROM = 3
AGE_TO = 34+1

AGE_FROM_M = AGE_FROM * 12
AGE_TO_M = AGE_TO * 12

#dense net settings
target_size_dense_net = [256,256] 

#unet settings
target_size_unet = [512,512] 
unet_classes = 2 #bg and segmentation

#for post processing
softmax_threshold = 0.5
major_voting = True
scaling_factor = 2.6
downsampling_factor = 1/scaling_factor