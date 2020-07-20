#Reference : https://github.com/yu4u/age-gender-estimation
import pandas as pd
import logging
import argparse
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from age_gender_model import WideResNet
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
from datetime import datetime
logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    args = parser.parse_args()
    return args

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

def load_data(mat_path):
    d = loadmat(mat_path)
    print(d)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]



def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    depth = args.depth
    k = args.width
    validation_split = args.validation_split
    use_augmentation = args.aug
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.debug("Loading data...")
    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    y_data_a = np_utils.to_categorical(age, 101)

    model = WideResNet(image_size, depth=depth, k=k)()
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

    logging.debug("Running training...")

    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    if use_augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255))
        training_generator = MixupGenerator(X_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2,
                                            datagen=datagen)()
        hist = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(X_test, [y_test_g, y_test_a]),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)
    else:
        hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
                         validation_data=(X_test, [y_test_g, y_test_a]))

    logging.debug("Saving history...")
    pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()