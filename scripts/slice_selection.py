import sys
sys.path.append('../TM2_segmentation')

import datetime
import json
import os
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


from scripts.generators import SliceSelectionSequence
from scripts.densenet_regression import DenseNet
from sklearn.model_selection import train_test_split
from settings import CUDA_VISIBLE_DEVICES

#  Train the model
def train(data_dir, model_dir, epochs=10, name=None, batch_size=16,
          load_weights=None, gpus=1, learning_rate=0.1, threshold=10.0,
          nb_layers_per_block=4, nb_blocks=4, nb_initial_filters=16,
          growth_rate=12, compression_rate=0.5, activation='relu',
          initializer='glorot_uniform', batch_norm=True, wandb_callback=True):

    args = locals()
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
   
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="slice-select-itmt2",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "activation": activation,
            
        }
    )


    # Set up dataset
    train_image_dir = os.path.join(data_dir, 'train')
    val_image_dir = os.path.join(data_dir, 'val')
    train_meta_file = os.path.join(data_dir,'train.csv')
    val_meta_file = os.path.join(data_dir, 'val.csv')
    train_labels = pd.read_csv(train_meta_file)['ZOffset'].values
    val_labels = pd.read_csv(val_meta_file)['ZOffset'].values
    print('\n\n\n','train_labels.shape:',train_labels.shape,'tuning_labels.shape:', val_labels.shape,'\n\n\n')
    train_jitter = 633  # default 1000 times of image augmentation for each epoch
    val_jitter = 181  # default 50 times of image augmentation for each epoch

    train_generator = SliceSelectionSequence(
        train_labels, train_image_dir, batch_size, train_jitter, jitter=True, sigmoid_scale=threshold
    )
    val_generator = SliceSelectionSequence(
        val_labels, val_image_dir, batch_size, val_jitter, sigmoid_scale=threshold
    )

    train_batches = train_labels.shape[0] // batch_size 
    print('\n \n  train_batches::::', train_labels.shape[0],train_batches, '\n')
    val_batches = val_labels.shape[0] // batch_size 
    print('\n \n  val_batches::::', val_labels.shape[0], val_batches, '\n')
    
    # Directories and files to use
    if name is None:
        name = 'untitled_model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(model_dir, name)
    tflow_dir = os.path.join(output_dir, 'tensorboard_log')
    weights_path = os.path.join(output_dir, 'weights-{epoch:02d}-{val_loss:.4f}.hdf5')
    architecture_path = os.path.join(output_dir, 'architecture.json')
    tensorboard = TensorBoard(log_dir=tflow_dir, histogram_freq=1, write_graph=True, write_images=True)

    with tf.device('/gpu:0'):
        model = DenseNet(
            img_dim=(256, 256, 1),
            nb_layers_per_block=nb_layers_per_block,
            nb_dense_block=nb_blocks,
            growth_rate=growth_rate,
            nb_initial_filters=nb_initial_filters,
            compression_rate=compression_rate,
            sigmoid_output_activation=True,
            activation_type=activation,
            initializer=initializer,
            output_dimension=1,
            batch_norm=batch_norm
        )
    if load_weights is None:
        os.mkdir(output_dir)
        os.mkdir(tflow_dir)

        args_path = os.path.join(output_dir, 'args.json')
        with open(args_path, 'w') as json_file:
            json.dump(args, json_file, indent=4)

        # Create the model
        print('Compiling model')
    # Save the architecture
        with open(architecture_path, 'w') as json_file:
            json_file.write(model.to_json())

    else:
        os.mkdir(output_dir)
        os.mkdir(tflow_dir)

        args_path = os.path.join(output_dir, 'args.json')
        with open(args_path, 'w') as json_file:
            json.dump(args, json_file, indent=4)

        # Create the model
        print('Loading model')
        # Save the architecture
        with open(architecture_path, 'w') as json_file:
            json_file.write(model.to_json())
        # Load the weights
        model.load_weights(load_weights)

    parallel_model = model
    keras_model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)

    RRc = ReduceLROnPlateau(monitor = "val_loss", 
                            factor = 0.5, 
                            patience = 3, 
                            min_lr=0.000001, 
                            verbose=1, 
                            mode='min')

    # Compile multi-gpu model
    loss = 'mean_squared_error'
    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    print('Starting training...')
    parallel_model.fit_generator(train_generator, train_batches,
                                 epochs=epochs,
                                 shuffle=True, 
                                 validation_data=val_generator,  validation_steps=val_batches,
                                 callbacks=[keras_model_checkpoint, 
                                            RRc,
                                            WandbMetricsLogger(log_freq=3)],
                                 use_multiprocessing=True,
                                 workers=64) 
    model.save_weights(os.path.join(output_dir, 'Top_Selection_Model_Weight.hdf5'))
    return model

