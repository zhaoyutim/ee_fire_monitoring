import tensorflow as tf
from tensorflow import keras

from model.vit.utilities.patch_encoder import PatchEncoder
from model.vit.utilities.patches import Patches
from tensorflow.keras import layers
import tensorflow_addons as tfa

class VisionTransformerGenerator:
    def __init__(self, train_dataset, input_shape, patch_size, resize_size, projection_dim, transformer_layers, num_heads, mlp_head_units, num_classes):
        # dataset should be N*h*l*c
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes
        self.resize_size = resize_size
        self.patch_size = patch_size

        # data augmentation and adaption on training data
        self.data_augmentation_layer = self.data_augmentation()
        self.data_augmentation_layer.layers[0].adapt(train_dataset)
        self.model = self.create_vit_classifier()


    def data_augmentation(self):
        data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(self.resize_size, self.resize_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        return data_augmentation


    def create_vit_classifier(self):
        inputs = layers.Input(shape=self.input_shape)
        # Augment data.
        augmented = self.data_augmentation_layer(inputs)
        # Create patches.
        patches = Patches(self.patch_size)(augmented)
        # Encode patches.
        num_patches = (self.resize_size // self.patch_size) ** 2
        encoded_patches = PatchEncoder(num_patches, self.projection_dim)(patches)
        transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP for classification.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def run_experiment(self, dataset_train, dataset_test, batch_size=256, num_epochs=100, learning_rate=0.001, weight_decay=0.0001):
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = self.model.fit(
            x=dataset_train[0],
            y=dataset_train[1],
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        self.model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = self.model.evaluate(dataset_test[0], dataset_test[1])
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return history