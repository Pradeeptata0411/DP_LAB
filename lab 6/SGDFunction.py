# preprocessing.py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

from keras.src import regularizers
from keras.src.layers import Conv2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD


class PreProcess_Data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img_name = train_class[j]
                img_path = os.path.join(dpath, i, img_name)
                img = cv2.imread(img_path)
                axs[count][j].set_title(i)
                axs[count][j].imshow(img)
            count += 1
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        imagefile = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                imagefile.append(img)
                label.append(i)
        print('Number of train images: {}\n'.format(len(imagefile)))
        print('Number of train image labels: {}\n'.format(len(label)))
        ret_df = pd.DataFrame({'Image': imagefile, 'Labels': label})
        return imagefile, label, ret_df

    def generate_train_test_images(self, imagefile, label):
        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        # Load images from file paths and convert them to numpy arrays
        x_train = [img_to_array(load_img(img_path, target_size=target_size)) for img_path in imagefile]
        x_train = np.array(x_train)

        tr_gen = train_datagen.flow(x_train, np.array(label), batch_size=32)

        # Similar placeholder code for test and validation generators

        return tr_gen, None, None


# Model.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class DeepANN:
    def CNN_MODEL(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model


    def CUSTOM_MODEL(self, input_shape, num_classes):
        model = Sequential()
        model.add(Dense(2, activation='relu', input_shape=input_shape))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model


# Your main script

import matplotlib.pyplot as plt

if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\deep learning\\Deep Learning Programs\\train'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")

    # Generate train, test, and validation generators
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(imagefile, label)

    print("train Generator :-", tr_gen)
    print("test Generator :-", tt_gen)
    print("validation Generator :-", va_gen)

    CnnModel = DeepANN()
    model1 = CnnModel.CUSTOM_MODEL(input_shape=(4,), num_classes=2)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model1.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    ANN_history = model1.fit(tr_gen, epochs=10, validation_data=va_gen)

    plt.plot(ANN_history.history['accuracy'], label='Training Accuracy')
    plt.plot(ANN_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(ANN_history.history['loss'], label='Training Loss')
    plt.plot(ANN_history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
