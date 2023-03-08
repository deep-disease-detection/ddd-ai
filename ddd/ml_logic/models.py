# import data processing libraries
import numpy as np

# import keras classes and functions
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.utils import to_categorical
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201


reg_kernel = l2(0.01)


def initialize_custom_model():
    '''
    Initialize the custom model
        Parameters:
            None
        Returns:
            model (keras.models.Model): initialized model
    '''
    # imput layer
    inputs = Input((256, 256, 1), name='input')

    # first batch of convolutions
    convolution_1 = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv1_1')(inputs)
    batch_convolution_1 = BatchNormalization()(convolution_1)
    convolution_2 = Conv2D(63, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv1_2')(batch_convolution_1)
    batch_convolution_2 = BatchNormalization()(convolution_2)
    concatenation_1 = concatenate([batch_convolution_2, inputs], axis=-1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(concatenation_1)
    drop_1 = Dropout(rate=0.2)(max_pool_1)

    # second batch of convolutions
    convolution_3 = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv2_1')(drop_1)
    batch_convolution_3 = BatchNormalization()(convolution_3)
    convolution_4 = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv2_2')(batch_convolution_3)
    batch_convolution_4 = BatchNormalization()(convolution_4)
    concatenation_2 = concatenate([batch_convolution_4, drop_1], axis=-1)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(concatenation_2)
    drop_2 = Dropout(rate=0.25)(max_pool_2)

    # third batch of convolutions
    convolution_5 = Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv3_1')(drop_2)
    batch_convolution_5 = BatchNormalization()(convolution_5)
    convolution_6 = Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv3_2')(batch_convolution_5)
    batch_convolution_6 = BatchNormalization()(convolution_6)
    concatenation_3 = concatenate([batch_convolution_6, drop_2], axis=-1)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(concatenation_3)
    drop_3 = Dropout(rate=0.30)(max_pool_3)

    # fourth batch of convolutions
    convolution_7 = Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv4_1')(drop_3)
    batch_convolution_7 = BatchNormalization()(convolution_7)
    convolution_8 = Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv4_2')(batch_convolution_7)
    batch_convolution_8 = BatchNormalization()(convolution_8)
    concatenation_4 = concatenate([batch_convolution_8, drop_3], axis=-1)
    max_pool_4 = MaxPooling2D(pool_size=(2, 2))(concatenation_4)
    drop_4 = Dropout(rate=0.35)(max_pool_4)

    # final batch of convolutions
    convolution_9 = Conv2D(256, (3, 3),
                           activation='relu',
                           padding='same',
                           kernel_regularizer=reg_kernel,
                           name='conv5_1')(drop_4)
    flatten_layer = Flatten()(convolution_9)
    dense_layer_1 = Dense(units=256, activation='relu')(flatten_layer)
    final_dropout_layer = Dropout(0.35)(dense_layer_1)
    dense_layer = Dense(units=256, activation='relu')(final_dropout_layer)
    output_layer = Dense(units=14, activation='softmax')(dense_layer)

    # model creation
    model = Model(inputs=[inputs], outputs=[output_layer])

    # model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_custom_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_validation: np.ndarray, y_validation: np.ndarray):
    '''
    Train the custom model with the given data
        Parameters:
            X_train (np.ndarray): training data
            y_train (np.ndarray): training labels
            X_validation (np.ndarray): validation data
            y_validation (np.ndarray): validation labels
        Returns:
            model (keras.models.Model): trained model
            history (keras.callbacks.History): training history

    '''
    model = initialize_custom_model()

    es = EarlyStopping(patience=5, verbose=2)

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_validation, y_validation),
                        callbacks=[es],
                        epochs=30,
                        batch_size=32,
                        verbose=1)

    return model, history


def VVG19_model():
    '''
    Create the VGG19 model
        Parameters:
            None
        Returns:
            model (keras.models.Model): VGG19 model
    '''
    # Load the pre-trained VGG19 model
    base_model = VGG19(weights='imagenet',
                       include_top=False,
                       input_shape=(256, 256, 3))

    # Freeze the layers in the base model to prevent them from being re-trained
    for layer in base_model.layers:
        layer.trainable = False

    # Add additional layers on top of the base model
    baseline_model = base_model.output
    flatten_layer = Flatten()(baseline_model)
    danse_layer_1 = Dense(1024, activation='relu')(flatten_layer)
    drop_layer_1 = Dropout(0.3)(danse_layer_1)
    danse_layer_2 = Dense(1024, activation='relu')(drop_layer_1)
    drop_layer_2 = Dropout(0.3)(danse_layer_2)
    output_layer = Dense(14, activation='softmax')(drop_layer_2)

    # Create the model by combining the base VGG19 model with the additional layers
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile the model with categorical_crossentropy loss function, adam optimizer and accuracy metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_VGG19_model(X_train: np.ndarray, y_train: np.ndarray,
                      X_validation: np.ndarray, y_validation: np.ndarray):
    '''
    Train the VGG19 model with the given data
        Parameters:
            X_train (np.ndarray): training data
            y_train (np.ndarray): training labels
            X_validation (np.ndarray): validation data
            y_validation (np.ndarray): validation labels
        Returns:
            model (keras.models.Model): trfained model
            history (keras.callbacks.History): training history

    '''
    model = VVG19_model()

    es = EarlyStopping(patience=5, verbose=2)

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_validation, y_validation),
                        callbacks=[es],
                        epochs=30,
                        batch_size=32,
                        verbose=1)

    return model, history





def initialize_DenseNet_model():
    """
    Create the DensetNet201 model
        Parameters:
            None
        Returns:
            model (keras.models.Model): DenseNet 201 model

    """
    base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=(256,256,3))

    # Set the first layers to be untrainable
    base_model.trainable = False

    add_layer = Dense(1024, activation='relu')
    drop_1 = Dropout(0.3)
    add_layer2 = Dense(1024, activation='relu')
    drop_2 = Dropout(0.3)
    flatten_layer = Flatten()
    dense_layer = Dense(256, activation='relu')
    prediction_layer = Dense(4, activation='softmax')

    model = Sequential([
            base_model,
            add_layer,
            drop_1,
            add_layer2,
            drop_2,
            flatten_layer,
            dense_layer,
            prediction_layer
        ])

    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    return model




def train_DenseNet_model(X_train, X_val):
    """
     Train the DN201 model with the given data
        Parameters:
            TBD
        Returns:
            model (keras.models.Model): trfained model
            history (keras.callbacks.History): training history
    """

    # Transform the images from grayscale to RGB in order to have 3 channels for the DN201
    X_train = X_train.map(lambda x,y : (tf.image.grayscale_to_rgb(x),y))
    X_val = X_val.map(lambda x,y : (tf.image.grayscale_to_rgb(x),y))


    model = initialize_DenseNet_model()
    es = EarlyStopping(patience=5, verbose=1)

    history = model.fit(X_train,
                    validation_data=X_val,
                    batch_size=32,
                    epochs=30,
                    verbose=1,
                    callbacks=[es])

    return model, history
