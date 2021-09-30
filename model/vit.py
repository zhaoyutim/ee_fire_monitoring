import numpy as np
import tensorflow as tf
from tensorflow import keras

from model.vit.utilities.patches import Patches
from model.vit.vit_model import VisionTransformerGenerator
import matplotlib.pyplot as plt


def visualizalize_patches():
    plt.figure(figsize=(4, 4))
    image = dataset_train[0][np.random.choice(range(dataset_train[0].shape[0]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

if __name__=='__main__':
    num_classes = 100

    dataset_train, dataset_test = keras.datasets.cifar100.load_data()

    print(f"x_train shape: {dataset_train[0].shape} - y_train shape: {dataset_train[1].shape}")
    print(f"x_test shape: {dataset_test[0].shape} - y_test shape: {dataset_test[1].shape}")
    input_shape = dataset_train[0][0,:,:,:].shape
    # Patch parameters
    image_size = 72  # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2

    # Transforer parameters
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8

    # Size of the dense layers of the final classifier
    mlp_head_units = [2048, 1024]

    visualizalize_patches()

    vit_gen = VisionTransformerGenerator(dataset_train[0], input_shape, patch_size, image_size, projection_dim, transformer_layers, num_heads, mlp_head_units, num_classes)

    history = vit_gen.run_experiment(dataset_train, dataset_test, batch_size=256, num_epochs=100, learning_rate=0.001, weight_decay=0.0001)