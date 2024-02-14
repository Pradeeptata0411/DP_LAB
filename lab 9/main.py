from preprocessing import PreProcess_Data

if __name__ == "__main__":
    images_folder_path = 'E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/cats_and_dog'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    tr_gen, tt_gen,va_gen = imdata.generate_train_test_images(imagefile, label)
    print("train Generator :-", tr_gen)


