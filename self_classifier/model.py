import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model


class CustomModel(Model):
    def call(self, x):
        return self.model(x)

    def reset(self):
        tf.keras.backend.clear_session()

    def prediction(self, x):
        """Function that takes softmax and argmax of the predictions"""
        predictions = self.model(x)
        pred_softmax = softmax(predictions, axis=0)
        top_class = tf.argmax(pred_softmax, axis=1)
        return top_class

    def to_prediction(self, predictions):
        """Function that takes softmax and argmax of the predictions"""
        top_class = tf.argmax(predictions, axis=1)
        return top_class


class DynamicModel(CustomModel):
    def __init__(self, input_shape, output_shape, n_layers=2, n_nodes=128,
                 activation=tf.keras.layers.ReLU(), batch_norm=True,
                  if_conv_layer=True, final_layer_bias=True, final_layer_activation=True):
        super(DynamicModel, self).__init__()
        # Input layers
        self.model = tf.keras.models.Sequential([tf.keras.Input(shape=input_shape)])
        if if_conv_layer:
          self.model.add(tf.keras.layers.Conv2D(32, 3, activation=activation))
        self.model.add(tf.keras.layers.Flatten())
        # Hidden Layers
        for _ in range(n_layers):
            self.model.add(tf.keras.layers.Dense(n_nodes))
            self.model.add(activation)
            if batch_norm:
                self.model.add(tf.keras.layers.BatchNormalization())

        # Output layer
        self.model.add(tf.keras.layers.Dense(output_shape, use_bias=final_layer_bias))
        if final_layer_activation:
            self.model.add(tf.keras.layers.Activation("softmax"))


class PretrainedModel(CustomModel):
    def __init__(self, pre_model, activation, input_shape, output_shape):
        super(PretrainedModel, self).__init__()

        if pre_model == "ResNet50":
            base_model = tf.keras.applications.resnet50.ResNet50(
                weights='imagenet', include_top=False,
                input_shape=input_shape)

        elif pre_model == "EfficientNetB0":
            base_model = tf.keras.applications.efficientnet.EfficientNetB0(
                weights='imagenet', include_top=False,
                input_shape=input_shape)

        elif pre_model == "EfficientNetB1":
            base_model = tf.keras.applications.efficientnet.EfficientNetB1(
                weights='imagenet', include_top=False,
                input_shape=input_shape)

        elif pre_model == "MobileNet":
            base_model = tf.keras.applications.mobilenet.MobileNet(
                weights='imagenet', include_top=False,
                input_shape=input_shape)

        for layer in base_model.layers:
            layer.trainable = False

        # Output layer
        x = tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation=activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation=activation)(x)
        predictions = tf.keras.layers.Dense(
            output_shape, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
