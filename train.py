import tensorflow as tf

from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import random
import copy

from enter_images import Image, mk_session


# interpret y label
def read_label(label):
    if label == "cat":
        return 1
    else:
        return 0

def label2array(label):
    as_int = read_label(label)
    return np.array(as_int)


# skeleton for future multi fold x-validation
def setup_validation_folds(length, n_folds):
    remainder = length % n_folds
    extra = 0
    out = []
    for i in range(n_folds):
        if i >= n_folds - remainder:
            extra = 1
        out += [i] * (length // n_folds + extra)
        
    random.shuffle(out)
    return out

def add_validation_folds(images, n_folds=4):
    random.shuffle(images)
    folds = iter(setup_validation_folds(len(images), n_folds))
    for img in images:
        img["fold"] = next(folds)


# db to list of dictionaries
def read_db(base_dir, db_path):
    sess, engine = mk_session(db_path)
    images = []
    for img in sess.query(Image):
        if not img.is_test_set:  # skip test images
            images.append({"x_path": "{}/{}/{}".format(
                base_dir, img.relative_path, img.file_name
            ),
                           "y_val": img.label})
    return images


# list of dictionaries to np. arrays
def input_list2arrays(list_in, y_function, which_fold=0):
    first_x = io.imread(list_in[0]["x_path"])
    first_y = y_function(list_in[0]["y_val"])

    out_x = np.zeros(shape=[len(list_in)] + list(first_x.shape))
    out_x[0] = first_x
    out_y = np.full(shape=[len(list_in)] + list(first_y.shape), fill_value=-1)
    out_y[0] = first_y
    for i in range(1, len(list_in), 1):
        out_x[i] = io.imread(list_in[i]["x_path"])
        out_y[i] = y_function(list_in[i]["y_val"])
    return out_x, out_y

def mk_mnist_model():
    # model basically from: https://www.tensorflow.org/alpha/tutorials/images/intro_to_cnns
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
              input_shape=(256, 192, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # finish on fc
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def main():
    image_paths_plus = read_db("raw_datas/", "localNSA.sqlite3")
    add_validation_folds(image_paths_plus)
    train_paths_plus = [x for x in image_paths_plus if x["fold"] != 0]
    val_paths_plus = [x for x in image_paths_plus if x["fold"] == 0]

    trainX, trainY = input_list2arrays(train_paths_plus, label2array)
    valX, valY = input_list2arrays(val_paths_plus, label2array)

    model = mk_mnist_model()

    model.compile(optimizer='adam',
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
    
    model.fit(trainX, trainY, epochs=1)

    print("final validtion")
    model.evaluate(valX, valY)


if __name__ == "__main__":
    main()
