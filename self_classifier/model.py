import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer, Rescaling,\
Dense, Flatten, BatchNormalization, Dropout, LayerNormalization, LeakyReLU, Lambda

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
        pred_softmax = softmax(predictions, axis=0)
        top_class = tf.argmax(pred_softmax, axis=1)
        return top_class


class DynamicModel(sc.CustomModel):
    def __init__(self, input_shape, n_layers=2, n_nodes=128, 
                 activation="relu", batch_norm=True):
        super(DynamicModel, self).__init__()
        # Input layers
        self.model = tf.keras.models.Sequential([
            InputLayer(input_shape=input_shape),
            Rescaling(1./255),
            Conv2D(16, 3, activation=activation),
            Flatten(),
            BatchNormalization(),
            Dropout(0.5)
            #tf.keras.layers.
        ])
        # Hidden Layers 
        for i in range(n_layers):
            self.model.add(Dense(n_nodes, activation=activation))
            if batch_norm:
                self.model.add(BatchNormalization())
                #self.model.add(tf.keras.layers.Dropout(0.25))

        # Output layer
        self.model.add(Dense(n_nodes))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        
        self.model.add(Lambda(lambda x: tf.math.l2_normalize(x)))
        self.model.add(Dense(10, use_bias=True))
 
 
class DynamicModelCnn(sc.CustomModel):
    def __init__(self, input_shape, n_classes=10, dropout=0.5):
        super(DynamicModelCnn, self).__init__()
        # Input layers
        model = tf.keras.models.Sequential()
        model.add(InputLayer(input_shape=input_shape)),
        model.add(Rescaling(1./255)),
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Conv2D(64, (3, 3), activation='relu'))

        # Output layer
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Lambda(lambda x: tf.math.l2_normalize(x)))
        model.add(Dense(n_classes, use_bias=True))
        
        self.model = model

class CNNModel(sc.CustomModel):
    def __init__(self, input_shape, activation="relu", batch_norm=True):
        super(CNNModel, self).__init__()
        models = tf.keras.models.Sequential()
        models.add(InputLayer(input_shape=input_shape))
        models.add(Rescaling(1./255))
        models.add(Conv2D(64, (5, 5),
                          padding="same",
                          activation="relu"))
        models.add(MaxPooling2D(pool_size=(2, 2)))
        models.add(Conv2D(128, (5, 5), padding="same",
                          activation="relu"))

        
        models.add(MaxPooling2D(pool_size=(2, 2)))
        models.add(Conv2D(256, (5, 5), padding="same",
                          activation="relu"))

        models.add(MaxPooling2D(pool_size=(2, 2)))
        models.add(Flatten())
        
        models.add(Dense(128))
        models.add(BatchNormalization())
        models.add(LeakyReLU())
        models.add(Dense(64))
        models.add(BatchNormalization())
        models.add(LeakyReLU())
        
        models.add(Lambda(lambda x: tf.math.l2_normalize(x)))
        models.add(Dense(10, use_bias=True))

        
        self.model = models