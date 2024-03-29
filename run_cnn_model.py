import argparse

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from segmentation_models.metrics import IOUScore, FScore
from segmentation_models.utils import set_trainable
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbCallback
from segmentation_models.losses import DiceLoss, BinaryCELoss
from segmentation_models import Unet, Linknet, PSPNet, FPN
import tensorflow.python.keras.backend as K
from model.swintransformer import SwinTransformer


def get_dateset(batch_size, data, nchannels):
    if data == 'palsar':
        train_array = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_train_'+str(nchannels)+'chan.npy')
        val_array = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_val_'+str(nchannels)+'chan.npy')
    else:
        train_array = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_train_'+str(nchannels)+'chan_s1.npy')
        val_array = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_val_'+str(nchannels)+'chan_s1.npy')
    print(train_array.shape)
    y_dataset = train_array[:,:,:,nchannels]>0
    y_dataset_val = val_array[:,:,:,nchannels]>0
    # x_train, x_val, y_train, y_val = train_test_split(train_dataset[:,:,:,:4], y_dataset, test_size=0.2, random_state=0)
    def make_generator(inputs, labels):
        def _generator():
            for input, label in zip(inputs, labels):
                yield input, label

        return _generator


    train_dataset = tf.data.Dataset.from_generator(make_generator(train_array[:,:,:,:nchannels], y_dataset),
                                                   (tf.float32, tf.float32))
    val_dataset = tf.data.Dataset.from_generator(make_generator(val_array[:,:,:,:nchannels], y_dataset_val),
                                                 (tf.float32, tf.float32))

    train_dataset = train_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)
    val_dataset = val_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)

    steps_per_epoch = train_array.shape[0]//batch_size
    validation_steps = val_array.shape[0]//batch_size

    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    smooth = 1.0
    return 1-(2.0*intersection+smooth)/(tf.math.reduce_sum(y_true_f)+tf.math.reduce_sum(y_pred_f)+smooth)

def masked_mse(mask_value):
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.sum(mask_true, axis=-1)
        return masked_mse
    f.__name__ = 'Masked MSE (mask_value={})'.format(mask_value)
    return f

def wandb_config(model_name, backbone, batch_size, learning_rate, data, nchannels):
    wandb.login()
    wandb.init(project='proj2_palsar', entity="zhaoyutim")
    wandb.run.name = 'model_name' + str(model_name) + 'backbone_'+ str(backbone)+ 'batchsize_'+str(batch_size)+'learning_rate_'+str(learning_rate)+'_data_'+str(data)+'_nchannels_'+str(nchannels)
    wandb.config = {
      "learning_rate": learning_rate,
      "weight_decay": weight_decay,
      "epochs": MAX_EPOCHS,
      "batch_size": batch_size,
      "model_name":model_name,
      "backbone": backbone
    }

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-p', type=str, help='Load trained weights')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-bb', type=str, help='backbone')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-data', type=str, help='dataset used')
    parser.add_argument('-nc', type=int, help='num of channels')
    args = parser.parse_args()
    model_name = args.m
    load_weights = args.p
    backbone = args.bb
    sm.set_framework('tf.keras')
    batch_size=args.b
    MAX_EPOCHS=300
    fine_tune=False
    learning_rate = args.lr
    data = args.data
    nchannels = args.nc
    weight_decay = learning_rate/10

    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset(batch_size, data, nchannels)

    wandb_config(model_name, backbone, batch_size, learning_rate, data, nchannels)

    strategy = tf.distribute.MirroredStrategy()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(256, 256),
        tf.keras.layers.Rescaling(1./255)
    ])
    with strategy.scope():
        if model_name == 'fpn':
            input = tf.keras.Input(shape=(None, None, nchannels))
            # x = resize_and_rescale(input)
            # x = data_augmentation(x)
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            basemodel = FPN(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1)
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)

        elif model_name == 'unet':
            input = tf.keras.Input(shape=(None, None, nchannels))
            # x = resize_and_rescale(input)
            # x = data_augmentation(x)
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            if backbone == 'None':
                basemodel = Unet(encoder_weights='imagenet', activation='sigmoid')
            else:
                basemodel = Unet(backbone, encoder_weights='imagenet', activation='sigmoid')
            basemodel.summary()
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)

        elif model_name == 'linknet':
            input = tf.keras.Input(shape=(None, None, nchannels))
            # x = resize_and_rescale(input)
            # x = data_augmentation(x)
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            basemodel = Linknet(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1)
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)

        elif model_name == 'pspnet':
            input = tf.keras.Input(shape=(None, None, nchannels))
            # x = resize_and_rescale(input)
            # x = data_augmentation(x)
            input_resize = tf.keras.layers.Resizing(384,384)(input)
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input_resize)
            basemodel = PSPNet(backbone, activation='sigmoid', classes=1)
            output = basemodel(conv1)
            output_resize = tf.keras.layers.Resizing(256,256)(output)
            model = tf.keras.Model(input, output_resize, name=model_name)
        elif model_name == 'swinunet':
            input = tf.keras.Input(shape=(256, 256, nchannels))
            input_resize = tf.keras.layers.Resizing(224,224)(input)
            # basemodel = models.swin_unet_2d((224, 224, 3), filter_num_begin=64, n_labels=1, depth=4, stack_num_down=2, stack_num_up=2,
            #                             patch_size=(2, 2), num_heads=[3, 6, 12, 24], window_size=[7, 7, 7, 7], num_mlp=512,
            #                             output_activation='Sigmoid', shift_window=True, name='swin_unet')
            basemodel = SwinTransformer('swin_tiny_224', num_classes=1, include_top=False, pretrained=False)

            # basemodel.summary()
            output = basemodel(input_resize)
            output_resize = tf.keras.layers.Resizing(256,256)(output)
            model = tf.keras.Model(input, output_resize, name=model_name)
        model.summary()
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        checkpoint = ModelCheckpoint('/geoinfo_vol1/zhao2/proj2_model/proj2_'+model_name+'_pretrained_'+backbone+'dataset_'+data, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        iou_score=IOUScore(threshold=0.5)
        f1_score=FScore(beta=1, threshold=0.5)
        binary_crossentropy = BinaryCELoss()
        dice_loss = DiceLoss()
        bce_dice_loss = binary_crossentropy + dice_loss
        masked_mse_loss = masked_mse(np.nan)
        model.compile(optimizer, loss=dice_coef, metrics=[iou_score, f1_score])

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)

    if load_weights== 'yes':
        model.load_weights('/geoinfo_vol1/zhao2/proj2_model/proj2_'+model_name+'_pretrained_'+backbone)
    else:
        # if fine_tune==True:
        #     for layer in model.layers:
        #         layer.trainable = False
        #     model.fit(
        #         train_dataset,
        #         batch_size=batch_size,
        #         steps_per_epoch=steps_per_epoch,
        #         validation_data=val_dataset,
        #         validation_steps=validation_steps,
        #         epochs=2,
        #         callbacks=[WandbCallback()],
        #     )
        # for layer in model.layers:
        #     layer.trainable = True
        # model.compile(optimizer, loss=bce_dice_loss, metrics=[iou_score, f1_score])
        print('training in progress')
        history = model.fit(
            train_dataset,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=MAX_EPOCHS,
            callbacks=[WandbCallback(), checkpoint],
        )
        model.save('/geoinfo_vol1/zhao2/proj2_model/proj2_'+model_name+'_pretrained_'+backbone+'_dataset_'+data+'_nchannels_'+str(nchannels))
