import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD


def visualization_images(dir_path, nimages):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    count = 0
    for i in os.listdir(dir_path):
        train_class = os.listdir(os.path.join(dir_path, i))
        for j in range(nimages):
            img_name = train_class[j]
            img_path = os.path.join(dir_path, i, img_name)
            img = cv2.imread(img_path)
            axs[count][j].set_title(i)
            axs[count][j].imshow(img)
        count += 1
    fig.tight_layout()
    plt.show(block=True)


def preprocess(dir_path):
    imagefile = []
    label = []
    for i in os.listdir(dir_path):
        train_class = os.listdir(os.path.join(dir_path, i))
        for j in train_class:
            img = os.path.join(dir_path, i, j)
            imagefile.append(img)
            label.append(i)
    print('Number of train images: {}\n'.format(len(imagefile)))
    print('Number of train image labels: {}\n'.format(len(label)))
    ret_df = pd.DataFrame({'Image': imagefile, 'Labels': label})
    return imagefile, label, ret_df


def generate_train_test_images(imagefile, label):
    project_df = pd.DataFrame({'Image': imagefile, 'Labels': label})
    train, test = train_test_split(project_df, test_size=0.1)

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, validation_split=0.15)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train,
        directory='./',
        x_col="Image",
        y_col="Labels",
        target_size=(32, 32),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=32,
        subset='training')

    test_generator = test_datagen.flow_from_dataframe(
        test,
        directory='./',
        x_col="Image",
        y_col="Labels",
        target_size=(32, 32),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=32,
        subset='validation')

    validation_generator = train_datagen.flow_from_dataframe(
        train,
        directory='./',
        x_col="Image",
        y_col="Labels",
        target_size=(32, 32),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=32,
        subset='validation')

    print(f"Train images shape: {train.shape}")
    print(f"Test images shape: {test.shape}")
    return train_generator, test_generator, validation_generator


def CNN_MODEL():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\imagefile'
    visualization_images(images_folder_path, 2)
    imagefile, label, df = preprocess(images_folder_path)
    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")

    tr_gen, tt_gen, va_gen = generate_train_test_images(imagefile, label)
    print("train Generator :-", tr_gen)
    print("test Generator :-", tt_gen)
    print("validation Generator :-", va_gen)

    model1 = CNN_MODEL()
    ANN_history = model1.fit(tr_gen, epochs=5, validation_data=va_gen)
    plt.plot(ANN_history.history['accuracy'], label='Training Accuracy')
    plt.plot(ANN_history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()
    Ann_test_loss, Ann_test_acc = model1.evaluate(tr_gen)
    print(f'Test accuracy: {Ann_test_acc}')
    print("The ANN architecture is ")
    print(model1.summary())
