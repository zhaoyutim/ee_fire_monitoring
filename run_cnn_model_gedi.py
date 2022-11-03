import argparse
import platform
import random

import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow.python.keras.backend as K
from segmentation_models import Unet, Linknet, PSPNet, FPN
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbCallback

import wandb


def set_global_seed(seed=21):
    # Tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)

def get_dateset_gedi(batch_size, nchannels):
    if platform.system() == 'Darwin':
        x_train = np.load('dataset/proj4_train_na2020train'+'.npy').astype(np.float32)
    else:
        x_train = np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_na2020train' + '.npy').astype(np.float32)
        # x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_sa' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_eu' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_au' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_af' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_sas' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_nas' + '.npy').astype(np.float32)), axis=0)
    y_train = x_train[:,:,:,10]
    if nchannels==6:
        x_train, x_val, y_train, y_val = train_test_split(np.nan_to_num(x_train[:, :, :, 3:9]), y_train,
                                                          test_size=0.2, random_state=0)
    else:
        x_train, x_val, y_train, y_val = train_test_split(np.nan_to_num(x_train[:,:,:,:nchannels]), y_train, test_size=0.2, random_state=0)
    def make_generator(inputs, labels):
        def _generator():
            for input, label in zip(inputs, labels):
                yield input, label

        return _generator

    train_dataset = tf.data.Dataset.from_generator(make_generator(x_train, y_train),
                                                   (tf.float32, tf.float32))
    val_dataset = tf.data.Dataset.from_generator(make_generator(x_val, y_val),
                                                 (tf.float32, tf.float32))

    train_dataset = train_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)
    val_dataset = val_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)

    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_train.shape[0]//batch_size

    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def masked_mse(y_true, y_pred):
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))
    mask_true = K.cast(K.not_equal(y_true, -1), K.floatx())
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.mean(K.sum(masked_squared_error, axis=-1) / (K.sum(mask_true, axis=-1) + K.epsilon()))
    return masked_mse

def masked_mae(y_true, y_pred):
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))
    mask_true = K.cast(K.not_equal(y_true, -1), K.floatx())
    mae = K.abs(mask_true * (y_true - y_pred))
    masked_mae = K.mean(K.sum(mae, axis=-1) / (K.sum(mask_true, axis=-1) + K.epsilon()))
    return masked_mae

def wandb_config(model_name, backbone, batch_size, learning_rate, nchannels):
    wandb.login()
    wandb.init(project='proj4_gedi', entity="zhaoyutim")
    wandb.run.name = 'model_name' + str(model_name) + 'backbone_'+ str(backbone)+ 'batchsize_'+str(batch_size)+'learning_rate_'+str(learning_rate)+'_nchannels_'+str(nchannels)
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": MAX_EPOCHS,
      "batch_size": batch_size,
      "model_name":model_name,
      "backbone": backbone
    }
def create_model(model_name, backbone, learning_rate, nchannels):
    if model_name == 'fpn':
        input = tf.keras.Input(shape=(None, None, 3))
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
        basemodel = FPN(backbone, encoder_weights='imagenet', activation='relu', classes=1)
        output = basemodel(conv1)
        model = tf.keras.Model(input, output, name=model_name)

    elif model_name == 'unet':
        input = tf.keras.Input(shape=(64, 64, nchannels))
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
        if backbone == 'None':
            basemodel = Unet(input_shape=(64, 64, 3), encoder_weights='imagenet', activation='relu')
        else:
            basemodel = Unet(backbone, input_shape=(64, 64, 3), encoder_weights='imagenet', activation='relu')
        output = basemodel(conv1)
        model = tf.keras.Model(input, output, name=model_name)

    elif model_name == 'linknet':
        input = tf.keras.Input(shape=(None, None, 3))
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
        basemodel = Linknet(backbone, encoder_weights='imagenet', activation='relu', classes=1)
        output = basemodel(conv1)
        model = tf.keras.Model(input, output, name=model_name)

    elif model_name == 'pspnet':
        input = tf.keras.Input(shape=(None, None, 3))
        input_resize = tf.keras.layers.Resizing(384,384)(input)
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input_resize)
        basemodel = PSPNet(backbone, activation='relu', classes=1)
        output = basemodel(conv1)
        output_resize = tf.keras.layers.Resizing(256,256)(output)
        model = tf.keras.Model(input, output_resize, name=model_name)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=masked_mse, metrics= masked_mse)
    return model

if __name__=='__main__':
    sm.set_framework('tf.keras')
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-bb', type=str, help='backbone')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-nc', type=int, help='num of channels')
    args = parser.parse_args()
    model_name = args.m
    backbone = args.bb
    batch_size=args.b
    learning_rate = args.lr
    nchannels = args.nc
    set_global_seed()

    model = create_model(model_name, backbone, learning_rate, nchannels)
    MAX_EPOCHS = 100
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset_gedi(batch_size, nchannels)
    wandb_config(model_name, backbone, batch_size, learning_rate, nchannels)
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)
    print('training in progress ')
    if platform.system() != 'Darwin':
        checkpoint = ModelCheckpoint(
            '/geoinfo_vol1/zhao2/proj4_model/proj4_' + model_name + '_pretrained_' + backbone,
            monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    else:
        checkpoint = ModelCheckpoint(
            'proj4_' + model_name + '_pretrained_' + backbone,
            monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    history = model.fit(
        train_dataset,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=MAX_EPOCHS,
        callbacks=[WandbCallback(), checkpoint],
    )
    if platform.system() != 'Darwin':
        model.save('/geoinfo_vol1/zhao2/proj4_model/proj4_'+model_name+'_pretrained_'+backbone+'_nchannels_'+str(nchannels))
    else:
        model.save('proj4_' + model_name + '_pretrained_' + backbone+'_nchannels_'+str(nchannels))
