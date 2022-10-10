import argparse
import random
import platform
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow.python.keras.backend as K
import wandb
from segmentation_models import Unet, Linknet, PSPNet, FPN
from segmentation_models.losses import DiceLoss, BinaryCELoss
from segmentation_models.metrics import IOUScore, FScore
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbCallback

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

def get_dateset_gedi(batch_size):
    if platform.system() == 'Darwin':
        x_train = np.load('dataset/proj4_train_na'+'.npy').astype(np.float32)
    else:
        x_train = np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_na' + '.npy').astype(np.float32)
        # x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_sa' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_eu' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_au' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_af' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_sas' + '.npy').astype(np.float32)), axis=0)
    #     x_train = np.concatenate((x_train, np.load('/geoinfo_vol1/zhao2/proj4_dataset/proj4_train_nas' + '.npy').astype(np.float32)), axis=0)
    y_train = x_train[:,:,:,8]
    x_train, x_val, y_train, y_val = train_test_split(np.nan_to_num(x_train[:,:,:,:3]), y_train, test_size=0.2, random_state=0)
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
    masked_mse = K.mean(K.sum(masked_squared_error, axis=-1) / K.sum(mask_true, axis=-1))
    return masked_mse

def masked_mae(y_true, y_pred):
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))
    mask_true = K.cast(K.not_equal(y_true, -1), K.floatx())
    mae = K.abs(mask_true * (y_true - y_pred))
    masked_mae = K.mean(K.sum(mae, axis=-1) / K.sum(mask_true, axis=-1))
    return masked_mae
# y_true = np.ones((16, 256, 256)).astype(np.float32)
# y_pred = np.zeros((16, 256, 256)).astype(np.float32)
# print(masked_mse(y_true, y_pred))

def wandb_config(model_name, backbone, batch_size, learning_rate):
    wandb.login()
    wandb.init(project='proj4_gedi', entity="zhaoyutim")
    wandb.run.name = 'model_name' + str(model_name) + 'backbone_'+ str(backbone)+ 'batchsize_'+str(batch_size)+'learning_rate_'+str(learning_rate)
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
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-bb', type=str, help='backbone')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-nc', type=int, help='num of channels')
    args = parser.parse_args()
    model_name = args.m
    backbone = args.bb
    sm.set_framework('tf.keras')
    batch_size=args.b
    MAX_EPOCHS=100
    fine_tune=False
    learning_rate = args.lr
    weight_decay = learning_rate/10
    set_global_seed()
    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset_gedi(batch_size)

    wandb_config(model_name, backbone, batch_size, learning_rate)

    # strategy = tf.distribute.MirroredStrategy()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(256, 256),
        tf.keras.layers.Rescaling(1./255)
    ])
    # with strategy.scope():
    if model_name == 'fpn':
        input = tf.keras.Input(shape=(None, None, 3))
        # x = resize_and_rescale(input)
        # x = data_augmentation(x)
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
        basemodel = FPN(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1)
        output = basemodel(conv1)
        model = tf.keras.Model(input, output, name=model_name)

    elif model_name == 'unet':
        input = tf.keras.Input(shape=(64, 64, 3))
        # x = resize_and_rescale(input)
        # x = data_augmentation(x)
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
        if backbone == 'None':
            basemodel = Unet(input_shape=(64, 64, 3), encoder_weights='imagenet', activation='linear')
        else:
            basemodel = Unet(backbone, input_shape=(64, 64, 3), encoder_weights='imagenet', activation='linear')
        output = basemodel(conv1)
        model = tf.keras.Model(input, output, name=model_name)

    elif model_name == 'linknet':
        input = tf.keras.Input(shape=(None, None, 3))
        # x = resize_and_rescale(input)
        # x = data_augmentation(x)
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input)
        basemodel = Linknet(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1)
        output = basemodel(conv1)
        model = tf.keras.Model(input, output, name=model_name)

    elif model_name == 'pspnet':
        input = tf.keras.Input(shape=(None, None, 3))
        # x = resize_and_rescale(input)
        # x = data_augmentation(x)
        input_resize = tf.keras.layers.Resizing(384,384)(input)
        conv1 = tf.keras.layers.Conv2D(3, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(input_resize)
        basemodel = PSPNet(backbone, activation='sigmoid', classes=1)
        output = basemodel(conv1)
        output_resize = tf.keras.layers.Resizing(256,256)(output)
        model = tf.keras.Model(input, output_resize, name=model_name)
    model.summary()
    if platform.system() != 'Darwin':
        checkpoint = ModelCheckpoint(
            '/geoinfo_vol1/zhao2/proj4_model/proj4_' + model_name + '_pretrained_' + backbone,
            monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    else:
        checkpoint = ModelCheckpoint(
            'proj4_' + model_name + '_pretrained_' + backbone,
            monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    iou_score=IOUScore(threshold=0.5)
    f1_score=FScore(beta=1, threshold=0.5)
    binary_crossentropy = BinaryCELoss()
    dice_loss = DiceLoss()
    bce_dice_loss = binary_crossentropy + dice_loss
    model.compile(optimizer, loss=masked_mae, metrics= masked_mse)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)

    print('training in progress ')
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
        model.save('/geoinfo_vol1/zhao2/proj4_model/proj4_'+model_name+'_pretrained_'+backbone)
    else:
        model.save('proj4_' + model_name + '_pretrained_' + backbone)
