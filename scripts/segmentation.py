import datetime
import json
import os
import numpy as np
import warnings

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from scripts.unet import get_unet_2D
from scripts.loss_unet import focal_tversky_loss_c
from scripts.generators import SegmentationSequence
from settings import target_size_unet,unet_classes, CUDA_VISIBLE_DEVICES

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
warnings.filterwarnings("ignore")
  
# Train the model  
def train(data_dir, model_dir, name=None, epochs=100, batch_size=1, load_weights=None,
          gpus=1, learning_rate=0.01, num_convs=2, activation='relu', 
          upsamlping_modules=5, initial_features=16):

    args = locals()   
    compression_channels = list(2**np.arange(int(np.log2(initial_features)),
                                             1+upsamlping_modules+int(np.log2(initial_features))))
    decompression_channels=sorted(compression_channels,reverse=True)[1:]
    
    # Start a run, tracking hyperparameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="seg-tmt",

        # track hyperparameters and run metadata with wandb.config
        config={
            "learning_rate": learning_rate,
            "epoch": epochs,
            "batch_size": batch_size,
            "activation": activation,
            "data_dir": data_dir,
            "model_dir": model_dir,
        }
    )

    # Load the data
    train_images_file = os.path.join(data_dir, 'train_images.npy')
    val_images_file = os.path.join(data_dir, 'val_images.npy')
    train_masks_file = os.path.join(data_dir, 'train_masks.npy')
    val_masks_file = os.path.join(data_dir, 'val_masks.npy')

    images_train = np.load(train_images_file)
    images_train = images_train.astype(float) #uint8
    images_val = np.load(val_images_file)
    images_val = images_val.astype(float)
    masks_train = np.load(train_masks_file)
    masks_train = masks_train.astype(np.uint8) #bool binary_crossentropy
    masks_val = np.load(val_masks_file)
    masks_val = masks_val.astype(np.uint8)
    print('\n\n\nimages_train.shape,images_val.shape', images_train.shape,images_val.shape,'\n\n\n')

    # Directories and files to use
    if name is None:
        name = 'untitled_model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(model_dir, name)
    tflow_dir = os.path.join(output_dir, 'tensorboard_log')
    os.mkdir(output_dir)
    os.mkdir(tflow_dir)
    weights_path = os.path.join(output_dir, 'weights-{epoch:02d}-{val_loss:.4f}.hdf5')
    architecture_path = os.path.join(output_dir, 'architecture.json')
    tensorboard = TensorBoard(log_dir=tflow_dir, histogram_freq=1, write_graph=True, write_images=True)

    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as json_file:
        json.dump(args, json_file, indent=4)

    print('Creating and compiling model...')
    with tf.device('/gpu:0'):
        model = get_unet_2D(
            unet_classes,
            (target_size_unet[0], target_size_unet[1], 1),
            num_convs=num_convs,
            activation=activation,
            compression_channels=compression_channels,
            decompression_channels=decompression_channels
            )

    # Save the architecture
    with open(architecture_path,'w') as json_file:
        json_file.write(model.to_json())

    # Use multiple devices
    if gpus > 1:
        parallel_model = multi_gpu_model(model, gpus)
    else:
        parallel_model = model

    # Should we load existing weights?
    if load_weights is not None:
        print('Loading pre-trained weights...')
        parallel_model.load_weights(load_weights)

    val_batches = images_val.shape[0] // batch_size 
    print('\n \n  val_batches::::',val_batches, '\n')
    train_batches = images_train.shape[0] // batch_size 
    print('\n \n  train_batches::::',train_batches, '\n')

    # Set up the learning rate scheduler
    def lr_func(e):
        if e > 0.75 * epochs:
            print("*1/4Lr ",0.2* learning_rate)
            return 0.01 * learning_rate
        elif e > 0.5 * epochs:
            print("*1/2Lr ",0.5* learning_rate)
            return 0.1 * learning_rate
        else:
            print("*Lr ",1 * learning_rate)
            return learning_rate
    # create a learning rate scheduler     
    lr_scheduler = LearningRateScheduler(lr_func)
    
    train_generator = SegmentationSequence(images_train, masks_train, batch_size, jitter=True)
    val_generator = SegmentationSequence(images_val, masks_val, batch_size, jitter=False)

    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=focal_tversky_loss_c)
    
    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=5) 
    
    RRc = ReduceLROnPlateau(monitor = "val_loss", 
                            factor = 0.5, 
                            patience = 3, 
                            min_lr=0.000001, 
                            verbose=1, 
                            mode='min')

    print('Fitting model...')

    keras_model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    
    parallel_model.fit_generator(train_generator, 
                                 train_batches, 
                                 epochs=epochs,
                                shuffle=True, 
                                validation_steps=val_batches,
                                validation_data=val_generator,
                                use_multiprocessing=True,               
                                workers=1, 
                                max_queue_size=40, 
                                callbacks=[keras_model_checkpoint,
                                            RRc, early,
                                            WandbMetricsLogger(log_freq=5)])     
    
    # Save the template model weights
    model.save_weights(os.path.join(output_dir, 'Top_Segmentation_Model_Weight.hdf5'))

    return model
