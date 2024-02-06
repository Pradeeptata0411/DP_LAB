import tensorflow as tf
from keras.src import regularizers
from keras.src.layers import Conv2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
from keras.applications import VGG16



class DeepANN():

    def simple_model(self, optimizer, num_classes):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def CNN_MODEL(self):
        model = Sequential()
        model.add(Conv2D(32,(3,3) , activation='relu' , input_shape=(28,28,3)))
        model.add(layers.MaxPooling2D(2,2))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model


    def cnn_reg_model(self , input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
        model.add(Dense(2,  activation='softmax', kernel_regularizer=regularizers.l1(0.01),bias_regularizer=regularizers.l1(0.01)))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model

    def vgg_cnn(self):
        model = Sequential()
        model.add(VGG16(weights='imagenet' ,  include_top=False,input_shape=(32,32,3)))
        model.add(Flatten())
        model.add(Dense(1024, activation=('relu')))
        model.add(Dense(512, activation=('relu')))
        model.add(Dense(256, activation=('relu')))
        model.add(Dense(128, activation=('relu')))
        model.add(Dense(2, activation=('softmax')))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model



