import argparse

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from wandb.integration.keras import WandbCallback
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score, f1_score
from segmentation_models import Unet, Linknet, PSPNet, FPN
from keras_unet_collection import models
from model.swintransformer import SwinTransformer


def get_dateset(batch_size):

    train_dataset = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_train_4chan.npy')
    # val_dataset = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_test.npy')
    print(train_dataset.shape)
    y_dataset = train_dataset[:,:,:,4]>0
    # y_dataset_val = val_dataset[:,:,:,3]>0
    x_train, x_val, y_train, y_val = train_test_split(train_dataset[:,:,:,:4], y_dataset, test_size=0.2, random_state=0)
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
    validation_steps = x_val.shape[0]//batch_size

    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    smooth = 1.0
    return 1-(2.0*intersection+smooth)/(tf.math.reduce_sum(y_true_f)+tf.math.reduce_sum(y_pred_f)+smooth)

def wandb_config(model_name, backbone):
    wandb.login()
    wandb.init(project='proj2_palsar', entity="zhaoyutim")
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
    args = parser.parse_args()
    model_name = args.m
    load_weights = args.p
    backbone = args.bb
    sm.set_framework('tf.keras')
    batch_size=args.b
    MAX_EPOCHS=100
    learning_rate = args.lr
    weight_decay = learning_rate/10

    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset(batch_size)

    wandb_config(model_name, backbone)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if model_name == 'fpn':
            input = tf.keras.Input(shape=(None, None, 4))
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            basemodel = FPN(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1)
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)

        elif model_name == 'unet':
            input = tf.keras.Input(shape=(None, None, 4))
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            if backbone == 'None':
                basemodel = Unet(encoder_weights='imagenet', activation='sigmoid')
            else:
                basemodel = Unet(backbone, encoder_weights='imagenet', activation='sigmoid')
            basemodel.summary()
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)

        elif model_name == 'linknet':
            input = tf.keras.Input(shape=(None, None, 4))
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            basemodel = Linknet(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1)
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)

        elif model_name == 'pspnet':
            input = tf.keras.Input(shape=(None, None, 4))
            input_resize = tf.keras.layers.Resizing(384,384)(input)
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input_resize)
            basemodel = PSPNet(backbone, activation='sigmoid', classes=1)
            output = basemodel(conv1)
            output_resize = tf.keras.layers.Resizing(256,256)(output)
            model = tf.keras.Model(input, output_resize, name=model_name)
        elif model_name == 'swinunet':
            input = tf.keras.Input(shape=(256, 256, 4))
            input_resize = tf.keras.layers.Resizing(224,224)(input)
            # basemodel = models.swin_unet_2d((224, 224, 3), filter_num_begin=64, n_labels=1, depth=4, stack_num_down=2, stack_num_up=2,
            #                             patch_size=(2, 2), num_heads=[3, 6, 12, 24], window_size=[7, 7, 7, 7], num_mlp=512,
            #                             output_activation='Sigmoid', shift_window=True, name='swin_unet')
            basemodel = SwinTransformer('swin_tiny_224', num_classes=1, include_top=False, pretrained=False)

            # basemodel.summary()
            output = basemodel(input_resize)
            output_resize = tf.keras.layers.Resizing(256,256)(output)
            model = tf.keras.Model(input, output_resize, name=model_name)
        elif model_name == 'transunet':
            input = tf.keras.Input(shape=(None, None, 4))
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            basemodel = models.transunet_2d((256, 256, 3), filter_num=[64, 128, 256, 512], n_labels=1, stack_num_down=2, stack_num_up=2,
                                        embed_dim=256, num_mlp=768, num_heads=3, num_transformer=12,
                                        activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid',
                                        batch_norm=True, pool=True, unpool='bilinear', name='transunet')
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)
        elif model_name == 'unet_2d':
            input = tf.keras.Input(shape=(None, None, 4))
            conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
            basemodel = models.unet_2d((None, None, 3), [64, 128, 256, 512, 1024], n_labels=1,
                                       stack_num_down=2, stack_num_up=1,
                                       activation='GELU', output_activation='Sigmoid',
                                       batch_norm=True, pool='max', unpool='nearest', name='unet')
            output = basemodel(conv1)
            model = tf.keras.Model(input, output, name=model_name)
        model.summary()

        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9
        )

        model.compile(optimizer, loss=bce_jaccard_loss, metrics=[iou_score, f1_score])

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)

    if load_weights== 'yes':
        model.load_weights('/geoinfo_vol1/zhao2/proj2_model/proj2_'+model_name+'_pretrained_'+backbone)
    else:
        print('training in progress')
        history = model.fit(
            train_dataset,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=MAX_EPOCHS,
            callbacks=[WandbCallback()],
        )
        model.save('/geoinfo_vol1/zhao2/proj2_model/proj2_'+model_name+'_pretrained_'+backbone)
