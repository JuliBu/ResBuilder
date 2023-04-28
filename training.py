"""In this file we have the main trianing-function. It is called in several other files like the pipeline.py."""

# Import Libraries
import random

import imageio
import keras
import pandas as pd
from keras.datasets import fashion_mnist, mnist, cifar10
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.layers import Conv2D, BatchNormalization, Concatenate
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Lambda
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import keras.backend as K
import visualkeras

from morph_net.network_regularizers import flop_regularizer
from morph_net.tools import structure_exporter

import os
import numpy as np
import cv2
import time
import shutil

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

# Import other homemade functions
import config
from config import regularization_parameters, training_parameters, environment_parameters, plot_options, input_parameters
from customFunctions import autoaugment
from customFunctions.changeArchitecture import gen_net_from_json_path, Conv2D_Layer, Pool_Layer
import customFunctions.labels
from zipfile import ZipFile
import scipy

import matplotlib.pyplot as plt

# A few Definitions of what is later needed in the training function
# Create specific datasets manual
def create_miniImageNet_data(trainOrTest, img_rows, img_cols):
    training_data = []
    training_label = []
    DATADIR = "/home/burghoff/kleineDatensaetze/ImageNetTest/"
    if trainOrTest == "train":
        DATADIR = os.path.join(DATADIR, "train")
    elif trainOrTest == "test":
        DATADIR = os.path.join(DATADIR, "val")
    CATEGORIES = ["ambulance", "balloon", "beach", "bird", "boat", "cow", "crane", "flower", "sheep", "tractor"]
    for category in CATEGORIES:  # do all categories
        path = os.path.join(DATADIR, category)  # create path to the categories
        class_num = CATEGORIES.index(category)  # get the classification  (0to 9). 0=ambulance, ...,  9=tractor
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (img_rows, img_cols))  # resize to normalize data size
                training_data.append(new_array)  # add this to our training_data
                training_label.append(class_num)
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            # except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    return [training_data, training_label]

def create_dataset_with_split(path_to_data_classes: str, img_rows: int, img_cols: int, percentage_train=0.8):
    x_train, x_test, y_train, y_test = [], [], [], []
    classes = os.listdir(path_to_data_classes)
    for class_index, one_class in enumerate(classes):
        print("Loading class " + str(one_class))
        x_values = []
        path_to_class_images = os.path.join(path_to_data_classes, str(one_class))
        list_image_names = os.listdir(path_to_class_images)
        for image_name in list_image_names:
            path_to_image = os.path.join(path_to_class_images, image_name)
            img_array = cv2.imread(path_to_image, cv2.IMREAD_COLOR)  # convert to array
            RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x_values.append(cv2.resize(RGB_img,(img_rows, img_cols))) # resize to normalize data size
        number_images_in_class = len(x_values)
        number_of_test_images = int(np.ceil(number_images_in_class * (1-percentage_train)))
        random.seed(42)
        list_of_test_indices = random.sample(range(number_images_in_class), number_of_test_images)
        for x_index, x_value in enumerate(x_values):
            if x_index in list_of_test_indices:
                x_test.append(x_value)
                y_test.append(class_index)
            else:
                x_train.append(x_value)
                y_train.append(class_index)
    return x_train, x_test, y_train, y_test


def create_custom_dataset(trainOrTest, dataset_name, img_rows, img_cols):
    # Define data folder
    data_folder = os.path.join(environment_parameters.custom_dataset_directory, dataset_name)
    if trainOrTest == "train":
        data_folder = os.path.join(data_folder, "Training", "Images")
    elif trainOrTest == "test":
        data_folder = os.path.join(data_folder, "Test", "Images")
    else:
        raise ValueError("trainOrTest varibale must be \"test\" or \"train\" and not" + str(trainOrTest))

    # Set categories
    if dataset_name == "GTSRB":
        CATEGORIES = []
        for cat_label in range(0,43):
            CATEGORIES.append(str(cat_label).zfill(5))
    else:
        raise ValueError("CATEGORIES for dataset " + dataset_name + "not defined!")

    # Load data for each category
    if dataset_name == "GTSRB":
        # German Traffic Signs Dataset
        if trainOrTest == "train":
            training_data = []
            training_label = []
            for category in CATEGORIES:
                class_path = os.path.join(data_folder, str(category))
                class_num = int(category)
                for img in tqdm(os.listdir(class_path)):  # iterate over each image per dogs and cats
                    try:
                        img_array = cv2.imread(os.path.join(class_path, img), cv2.IMREAD_COLOR)  # convert to array
                        RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        new_array = cv2.resize(RGB_img, (img_rows, img_cols))  # resize to normalize data size
                        training_data.append(new_array)  # add this to our training_data
                        training_label.append(class_num)
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
            return [training_data, training_label]
        elif trainOrTest == "test":
            test_data = []
            test_label = []
            test_files_path = data_folder
            gt_data = pd.read_csv(os.path.join(data_folder, "GT-final_test.csv"), sep=';')
            for img in tqdm(os.listdir(test_files_path)):  # iterate over each image per dogs and cats
                try:
                    img_array = cv2.imread(os.path.join(test_files_path, img), cv2.IMREAD_COLOR)  # convert to array
                    new_array = cv2.resize(img_array, (img_rows, img_cols))  # resize to normalize data size
                    test_data.append(new_array)  # add this to our training_data
                    class_num = gt_data[(gt_data["Filename"] == img)]["ClassId"].values[0]
                    test_label.append(class_num)
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
            return [test_data, test_label]
        else:
            raise ValueError("trainOrTest varibale must be \"test\" or \"train\" and not" + str(trainOrTest))
    else:
        raise ValueError("Create_custom_dataset does not support " + str(dataset_name))

def create_tiny_imagenet(path_to_zip_file="/home/burghoff/kleineDatensaetze/tiny-imagenet-200.zip"):
    """
    Tiny Imagenet has 200 classes. Each class has 500 training images, 50
    validation images, and 50 test images. We have released the training and
    validation sets with images and annotations. We provide both class labels an
    bounding boxes as annotations; however, you are asked only to predict the
    class label of each image without localizing the objects. The test set is
    released without labels. You can download the whole tiny ImageNet dataset
    here.
    """

    # Loading the file
    f = ZipFile(path_to_zip_file, "r")
    names = [name for name in f.namelist() if name.endswith("JPEG")]
    val_classes = np.loadtxt(
        f.open("tiny-imagenet-200/val/val_annotations.txt"),
        dtype=str,
        delimiter="\t",
    )
    val_classes = dict([(a, b) for a, b in zip(val_classes[:, 0], val_classes[:, 1])])
    x_train, x_test, x_valid, y_train, y_test, y_valid = [], [], [], [], [], []
    for name in names:
        if "train" in name:
            classe = name.split("/")[-1].split("_")[0]
            x_train.append(
                imageio.imread(f.open(name).read(), pilmode="RGB").transpose(
                    (2, 0, 1)
                )
            )
            y_train.append(classe)
        if "val" in name:
            x_valid.append(
                imageio.imread(f.open(name).read(), pilmode="RGB").transpose(
                    (2, 0, 1)
                )
            )
            arg = name.split("/")[-1]
            #print(val_classes[arg])
            y_valid.append(val_classes[arg])
        if "test" in name:
            x_test.append(
                imageio.imread(f.open(name).read(), pilmode="RGB").transpose(
                    (2, 0, 1)
                )
            )
    return x_train, x_valid, y_train, y_valid
class Dataset:
    """Saves the meta information of a dataset"""
    def __init__(self, name: str):
        self.name = name
        if name == "FashionMNIST":
            self.rows = 28
            self.columns = 28
            self.colors = 1
            self.data = fashion_mnist.load_data()
            self.batch_size = 128
        elif name == "EMNIST":
            self.rows = 28
            self.columns = 28
            self.colors = 1
            self.batch_size = 128
            emnist_train = tfds.load('emnist', split=tfds.Split.TRAIN, batch_size=-1)
            emnist_test = tfds.load('emnist', split=tfds.Split.TEST, batch_size=-1)
            emnist_train_data = tfds.as_numpy(emnist_train)
            emnist_test_data = tfds.as_numpy(emnist_test)
            self.data = [emnist_train_data["image"], emnist_train_data["label"], emnist_test_data["image"],
                         emnist_test_data["label"]]
        elif name == "MNIST":
            self.rows = 28
            self.columns = 28
            self.colors = 1
            self.data = mnist.load_data()
            self.batch_size = 128
        elif name == "CIFAR10":
            self.rows = 32
            self.columns = 32
            self.colors = 3
            self.data = cifar10.load_data()
            self.batch_size = 128
        elif name == "miniImageNet":
            self.rows = 224
            self.columns = 224
            self.colors = 3
            train_data = create_miniImageNet_data(trainOrTest="train", img_rows=224, img_cols=224)
            test_data = create_miniImageNet_data(trainOrTest="test", img_rows=224, img_cols=224)
            self.data = [train_data, test_data]
            self.batch_size = 16
        elif name == "TinyImageNet":
            self.rows = 64
            self.columns = 64
            self.colors = 3
            x_train, x_valid, y_train, y_valid = create_tiny_imagenet()
            self.data = [x_train, y_train, x_valid, y_valid]
            self.batch_size = 32
        elif name == "GTSRB":
            self.rows = 20
            self.columns =  20
            self.colors = 3
            train_data = create_custom_dataset(trainOrTest="train", dataset_name="GTSRB", img_rows=self.rows, img_cols=self.columns)
            test_data = create_custom_dataset(trainOrTest="test", dataset_name="GTSRB", img_rows=self.rows, img_cols=self.columns)
            self.data = [train_data, test_data]
            self.batch_size = 128
        elif name == "smallNORB":
            self.rows = 96
            self.columns = 96
            self.colors = 1
            self.batch_size = 64
            norb_train = tfds.load('smallnorb', split=tfds.Split.TRAIN, batch_size=-1)
            norb_test = tfds.load('smallnorb', split=tfds.Split.TEST, batch_size=-1)
            norb_train_data = tfds.as_numpy(norb_train)
            norb_test_data = tfds.as_numpy(norb_test)
            self.data = [norb_train_data["image"], norb_train_data["label_category"], norb_test_data["image"], norb_test_data["label_category"]]
        elif name == "CIFARhundred":
            self.rows = 32
            self.columns = 32
            self.colors = 3
            self.batch_size = 128
            cifarH_train = tfds.load('cifar100', split=tfds.Split.TRAIN, batch_size=-1)
            cifarH_test = tfds.load('cifar100', split=tfds.Split.TEST, batch_size=-1)
            cifarH_train_data = tfds.as_numpy(cifarH_train)
            cifarH_test_data = tfds.as_numpy(cifarH_test)
            self.data = [cifarH_train_data["image"], cifarH_train_data["label"], cifarH_test_data["image"],
                         cifarH_test_data["label"]]
        elif name == "Animals10":
            self.rows = 300
            self.columns = 300
            self.colors = 3
            self.batch_size = 16
            self.data = create_dataset_with_split("/home/burghoff/kleineDatensaetze/Animals10", self.rows, self.columns)

def get_dataset_information(name_of_dataset):
    # if dataset = Dataset(name_of_dataset)
    # return dataset
    # else raise Error
    if name_of_dataset == "FashionMNIST":
        return Dataset("FashionMNIST")
    elif name_of_dataset == "MNIST":
        return Dataset("MNIST")
    elif name_of_dataset == "CIFAR10":
        return Dataset("CIFAR10")
    elif name_of_dataset == "miniImageNet":
        return Dataset("miniImageNet")
    elif name_of_dataset == "smallNORB":
        return Dataset("smallNORB")
    elif name_of_dataset == "EMNIST":
        return Dataset("EMNIST")
    elif name_of_dataset == "CIFARhundred":
        return Dataset("CIFARhundred")
    elif name_of_dataset =="GTSRB":
        return Dataset("GTSRB")
    elif name_of_dataset =="TinyImageNet":
        return Dataset("TinyImageNet")
    elif name_of_dataset == "Animals10":
        return Dataset("Animals10")
    else:
        raise ValueError("Dataset \"" + str(name_of_dataset) + "\" not found!")

def gen_complex_data_augmentation(dataset=None):
    if dataset == "GTSRB":
        rotation_range = 0
        width_shift_range = 0.25
        height_shift_range = 0.25
        horizontal_flip = False
        vertical_flip = False
    else:
        # Set default values for data augmentation
        rotation_range = 0
        width_shift_range = 0.1
        height_shift_range = 0.1
        horizontal_flip = True
        vertical_flip = False
    image_gen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=rotation_range,
                # randomly shift images horizontally
                width_shift_range=width_shift_range,
                # randomly shift images vertically
                height_shift_range=height_shift_range,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=horizontal_flip,
                # randomly flip images
                vertical_flip=vertical_flip,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
    return image_gen

def gen_simple_data_augmentation():
    policy = autoaugment.CIFAR10Policy()
    def policy_wrapper(x):
        return np.asarray(policy(Image.fromarray(np.asarray(x, dtype="uint8"))))

    image_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1,
                                 horizontal_flip=True, fill_mode="constant", cval=128,
                                 preprocessing_function=policy_wrapper, dtype="uint8")
    return image_gen

def run_training(number_of_run: int, epochs: int,
                # Tensorboard Outputfile Options - histograms will slow down training
                generate_tb_files=False, histograms=0,
                # MorphNet parameters
                export_morphNet_structure=False,
                # Import Weights?
                importWeights=False,
                # All Layers trainable or freeze the new one?
                old_Layers_trainable=True,
                # Stop Early?
                early_stopping=False,
                # name of dataset
                name_of_data = "FashionMNIST",
                 # Path to net
                path_to_net = "/home/burghoff/Daten/200731_startnetze/03layernet.json",
                # Dummy Training
                dummy_training = False,
                # Train without regularization for testing accuracy of final net
                train_without_reg = False,
                # Start net if a new folder is generated
                start_net_json_file_name = environment_parameters.startNet_name
                ):
    # Todo: Tidy up working directories
    DEFAULT_DIR = os.path.join(environment_parameters.working_directory, name_of_data)
    path_to_tb_file = os.path.join(DEFAULT_DIR, "tb_outputs", str(number_of_run))
    OUTPUT_DIR1 = DEFAULT_DIR
    K.clear_session()

    # Print some parameters to stdout
    def print_attribute(var, string):
        print(string + ": " + str(var))

    print("Doing the run with these parameters:")
    print_attribute(number_of_run, "number_of_run")
    print_attribute(epochs, "epochs")
    print_attribute(DEFAULT_DIR, "DEFAULT_DIR")
    print_attribute(environment_parameters.default_gpu_index, "gpu_index")
    print_attribute(environment_parameters.gpu_fragment_default, "gpu_memory_fragment")
    print_attribute(generate_tb_files, "generate_tb_files")
    print_attribute(histograms, "histograms")
    print_attribute(path_to_tb_file, "path_to_tb_file")
    print_attribute(regularization_parameters.morphNet_Lambda, "morphNet_Lambda")
    print_attribute(export_morphNet_structure, "export_morphNet_structure")
    print_attribute(importWeights, "importWeights")
    print_attribute(old_Layers_trainable, "old_Layers_trainable")
    print_attribute(name_of_data, "name_of_data")
    print_attribute(path_to_net, "path_to_net")
    print_attribute(dummy_training, "dummy_training")
    print_attribute(train_without_reg, "train_without_reg")

    start_timer = time.time()

    # Set up GPU
    tf_config = tf.ConfigProto()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(environment_parameters.default_gpu_index)
    tf_config.gpu_options.per_process_gpu_memory_fraction = environment_parameters.gpu_fragment_default
    # tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.allow_growth = environment_parameters.allow_gpu_growth

    set_session(tf.Session(config=tf_config))

    default_acc_output = True

    # Morphnet Regularization Strength can be changed as a parameter in the function call
    if train_without_reg:
        morphRegulStrength = 0
        default_acc_output = False
    else:
        morphRegulStrength = regularization_parameters.morphNet_Lambda
    morphThreshold = 1e-2

    dataset = get_dataset_information(name_of_data)

    # Has to be the same as the number of classes of the used data
    num_classes = 10 #ToDo DoC

    # ToDo DoC Dataset Klasse einheitlich gestalten
    # Split data in training and test data
    if "MNIST" == dataset.name or "FashionMNIST" == dataset.name or "CIFAR10" == dataset.name:
        (x_train, y_train), (x_test, y_test) = dataset.data
        img_rows = dataset.rows
        img_cols = dataset.columns
    elif "miniImageNet" in dataset.name:
        x_train = np.array(dataset.data[0][0])
        y_train = np.array(dataset.data[0][1])
        x_test = np.array(dataset.data[1][0])
        y_test = np.array(dataset.data[1][1])
    elif "smallNORB" in dataset.name or "EMNIST" == dataset.name:
        x_train = np.array(dataset.data[0])
        y_train = np.array(dataset.data[1])
        x_test = np.array(dataset.data[2])
        y_test = np.array(dataset.data[3])
        if "EMNIST" == dataset.name:
            num_classes = 62
    elif "CIFARhundred" in dataset.name:
        x_train = np.array(dataset.data[0])
        y_train = np.array(dataset.data[1])
        x_test = np.array(dataset.data[2])
        y_test = np.array(dataset.data[3])
        num_classes = 100
    elif "GTSRB" in dataset.name:
        x_train = np.array(dataset.data[0][0])
        y_train = np.array(dataset.data[0][1])
        x_test = np.array(dataset.data[1][0])
        y_test = np.array(dataset.data[1][1])
        num_classes = 43
    elif "TinyImageNet" in dataset.name:
        x_train = np.array(dataset.data[0])
        y_train = np.array(dataset.data[1])
        x_test = np.array(dataset.data[2])
        y_test = np.array(dataset.data[3])
        num_classes = 200
    elif "Animals10" in dataset.name:
        x_train = np.array(dataset.data[0])
        y_train = np.array(dataset.data[2])
        x_test = np.array(dataset.data[1])
        y_test = np.array(dataset.data[3])
        num_classes = 10
    else:
        raise ValueError("The dataset " + dataset.name + " is not supported by the run_training function in training.py.")

    # if there is no startnet genereate it
    if not os.path.exists(path_to_net):
        os.makedirs(os.path.join(os.path.dirname(path_to_net)), exist_ok=True)
        start_net_json_file = os.path.join(environment_parameters.startNet_folder, start_net_json_file_name)
        shutil.copyfile(start_net_json_file, path_to_net)

    OUTPUT_DIR2 = OUTPUT_DIR1

    # Have a look on channels_first or channels_last because its important for MorphNet
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], dataset.colors, dataset.rows, dataset.columns)
        x_test = x_test.reshape(x_test.shape[0], dataset.colors, dataset.rows, dataset.columns)
        #input_shape = (dataset.colors, dataset.rows, dataset.cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], dataset.rows, dataset.columns, dataset.colors)
        x_test = x_test.reshape(x_test.shape[0], dataset.rows, dataset.columns, dataset.colors)
        #input_shape = (dataset.rows, dataset.cols, dataset.colors)

    # calculate steps_per_epoch
    number_of_training_images = x_train.shape[0]
    steps_per_epoch = int(number_of_training_images/dataset.batch_size)

    # Reshape the dataset to fit to the net
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    def CIFAR10_color_preprocessing(x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        return x_train, x_test

    def generic_color_preprocessing(x_train, x_test, x_val):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        for i in range(3):
            # ToDo: Calculate mean/std only over training set or complete dataset?
            this_mean = np.mean(x_train[:, :, :, i])
            this_std = np.std(x_train[:, :, :, i])
            x_train[:, :, :, i] = (x_train[:, :, :, i] - this_mean) / this_std
            x_test[:, :, :, i] = (x_test[:, :, :, i] - this_mean) / this_std
            x_val[:, :, :, i] = (x_val[:, :, :, i] - this_mean) / this_std
        return x_train, x_test, x_val

    # ToDo have a close look if correct
    def MNIST_color_preprocessing(x_train, x_test):
        x_train /= 255
        x_test /= 255
        return x_train, x_test

    # Do a split to generate val data
    if config.input_parameters.split_a_seperate_val_set:
        def create_val_split(x_train_old, y_train_old, percentage_train=0.8):
            x_train_stay, x_new_val, y_train_stay, y_new_val = [], [], [], []
            number_of_test_images = int(np.ceil(x_train_old.shape[0] * (1-percentage_train)))
            random.seed(42)
            list_of_test_indices = random.sample(range(x_train_old.shape[0]), number_of_test_images)
            for x_index, x_value in enumerate(x_train_old):
                if x_index in list_of_test_indices:
                    x_new_val.append(x_value)
                    y_new_val.append(y_train_old[x_index])
                else:
                    x_train_stay.append(x_value)
                    y_train_stay.append(y_train_old[x_index])
            return x_train_stay, x_new_val, y_train_stay, y_new_val
        
        x_train, x_val, y_train, y_val = create_val_split(x_train, y_train)
    else:
        x_val = x_test
        y_val = y_test

    # Do data preprocessing
    if config.training_parameters.do_data_preprocessing:
        if name_of_data == "CIFAR10":
            x_train, x_test = CIFAR10_color_preprocessing(x_train, x_test)
            # elif name_of_data == "FashionMNIST":
            #    x_train, x_test = MNIST_color_preprocessing(x_train, x_test)
        elif "MNIST" in name_of_data:
            # ToDo: Skipped Data preprocessing for any MNIST dataset
            x_train, x_test = x_train, x_test
        else:
            x_train, x_test, x_val = generic_color_preprocessing(x_train, x_test, x_val)

    
    


    # convert class vectors to binary class matrices
    if dataset.name == "TinyImageNet":
        # y_train = [y_el[1:] for y_el in y_train]
        # y_test = [y_el2[1:] for y_el2 in y_test]
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if config.output_parameters.show_images_before_aug:
        def plot_image_before_aug(image_index, number_in_row, n_images):
            plt.subplot(1, n_images, number_in_row + 1)
            tmp_image_index = image_index
            if "MNIST" in dataset.name or dataset.name == "smallNORB":
                # plt.imshow(np.uint8(x_train[tmp_image_index].squeeze()))
                plt.imshow(np.uint8(x_train[tmp_image_index].reshape((96, 96))), cmap='gray')
            else:
                plt.imshow(np.uint8(x_train[tmp_image_index]))
            coarse_label = customFunctions.labels.get_coarse_labels(name_of_data)
            label_index = -1
            for running_index, tmp_label_index in enumerate(y_train[tmp_image_index]):
                if tmp_label_index == 1:
                    label_index = int(running_index)
                    break
            plt.title(coarse_label[label_index])

        def show_a_few_images_before_aug():
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
            #                    hspace=0, wspace=0)
            # plt.margins(0,0)
            image_indices = config.output_parameters.image_indices
            for no_in_row, image_index in enumerate(image_indices):
                plot_image_before_aug(image_index, no_in_row, len(image_indices))
                plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR2, 'example_data.png'), bbox_inches='tight', figsize=(6, 2))
            plt.show(bbox_inches='tight', figsize=(6, 2))

    if config.output_parameters.show_images_before_aug:
        show_a_few_images_before_aug()

    # Should Data Augmentation be done?
    do_data_augmentation = training_parameters.do_training_with_data_augmentation

    # Data Augmentation
    if do_data_augmentation:
        if training_parameters.use_complex_data_aug:
            datagen = gen_complex_data_augmentation(dataset.name)
        else:
            datagen = gen_simple_data_augmentation()
        datagen.fit(x_train)
        tmp_iterator = datagen.flow(x_train, y=y_train, batch_size=dataset.batch_size)

        if config.output_parameters.show_images_after_aug :
            def plot_image_after_aug(image_index, number_in_row, image_indices):
                n_images = len(image_indices)
                augment_show_iterator = datagen.flow(x_train, y=y_train, batch_size=max(image_indices), shuffle=False)
                aug_batch = augment_show_iterator.next()
                images = aug_batch[0]
                plt.subplot(1, n_images, number_in_row + 1)
                tmp_image_index = image_index
                if "MNIST" in dataset.name or dataset.name == "smallNORB":
                    # plt.imshow(np.uint8(x_train[tmp_image_index].squeeze()))

                    plt.imshow(np.uint8(images[tmp_image_index].reshape((96, 96))), cmap='gray')
                else:
                    plt.imshow(np.uint8(images[tmp_image_index]))
                coarse_label = customFunctions.labels.get_coarse_labels(name_of_data)
                label_index = -1
                for running_index, tmp_label_index in enumerate(y_train[tmp_image_index]):
                    if tmp_label_index == 1:
                        label_index = int(running_index)
                        break
                plt.title(coarse_label[label_index])

            def show_a_few_images_after_aug():
                # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                #                    hspace=0, wspace=0)
                # plt.margins(0,0)
                image_indices = config.output_parameters.image_indices
                n_images = len(image_indices)
                augment_show_iterator = datagen.flow(x_train, y=y_train, batch_size=max(image_indices)+1, shuffle=False)
                aug_batch = augment_show_iterator.next()
                images = aug_batch[0]
                for no_in_row, image_index in enumerate(image_indices):
                    #plot_image_after_aug(image_index, no_in_row, image_indices)
                    plt.subplot(1, n_images, no_in_row + 1)
                    if "MNIST" in dataset.name or dataset.name == "smallNORB":
                        # plt.imshow(np.uint8(x_train[tmp_image_index].squeeze()))
                        plt.imshow(np.uint8(images[image_index].reshape((96, 96))), cmap='gray')
                    else:
                        plt.imshow(np.uint8(images[image_index]))
                    plt.axis('off')
                plt.savefig(os.path.join(OUTPUT_DIR2, 'example_data_aug.png'), bbox_inches='tight', figsize=(6, 2))
                plt.show(bbox_inches='tight', figsize=(6, 2))
            show_a_few_images_after_aug()
    # Define functions for generating Conv Layer

    def gen_conv_layer_new_format(layer:Conv2D_Layer, trainable=True, kernel_init='glorot_uniform', kernel_reg_str=None):
        if dummy_training:
            trainable = False
        if config.regularization_parameters.do_additional_l2_reg:
            kernel_reg =tf.keras.regularizers.L1L2(l1=regularization_parameters.layer_lasso_reg_strength,
                                                   l2=regularization_parameters.l2_regularization)
        elif kernel_reg_str > 0:
            kernel_reg = tf.keras.regularizers.L1L2(l1=regularization_parameters.layer_lasso_reg_strength, l2=0.0)
        else:
            kernel_reg = None
        if training_parameters.ignore_strides_in_convs:
            tmp_strides = (1,1)
        else:
            tmp_strides = (layer.stride, layer.stride)
        return Conv2D(layer.channel_number, (layer.filter_size_x, layer.filter_size_y),
                      strides=tmp_strides,
                      padding='same',
                      data_format='channels_last',
                      dilation_rate=(1, 1),
                      activation='relu', use_bias=True,
                      kernel_initializer=kernel_init,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_reg, bias_regularizer=None, activity_regularizer=None,
                      kernel_constraint=None, bias_constraint=None, name=layer.name, trainable=trainable)

    # Load the net including the skip connections
    net = gen_net_from_json_path(path_to_net)
    layer_list =net.layer_list
    skip_connections = net.skip_connect_list

    # Set flattened to seperate Conv-Layer part from Dense-Layer part of the net
    flattened = False

    # Build the model
    inputs = Input(shape=(x_train.shape[1],x_train.shape[2], x_train.shape[3]))
    x = inputs
    for layer in layer_list:
        # Build Conv Layers
        if layer.layer_type == "Conv2D":
            # look for starting skip connections
            for skip_connection in skip_connections:
                if skip_connection.start_layer == layer:
                    # Save the value of x to add it later back to the net
                    skip_connection.skip_value = x

            # Initiate Layers which have been added during the training
            if "New_Conv2D" in layer.name:
                if train_without_reg:
                    kernel_reg_str = 0
                else:
                    kernel_reg_str = regularization_parameters.layer_lasso_reg_strength
                if training_parameters.do_batch_normalization:
                    x = BatchNormalization(name=layer.name[0:4] + "BatchNorm"+layer.name[10:])(x)
                x = gen_conv_layer_new_format(layer, kernel_reg_str=kernel_reg_str)(x)
            else:
                if training_parameters.do_batch_normalization:
                    x = BatchNormalization(name=layer.name[0:5] + "BatchNorm" + layer.name[11:])(x)
                x = gen_conv_layer_new_format(layer)(x)

            # Look for ending skip_connections, so those end AFTER the layer
            for skip_connection in skip_connections:
                if skip_connection.target_layer == layer:
                    # In this case the skip connection ends in this layer.
                    # So first we have a look if the dimensions of the current x and the skip_value are the same.
                    default_path_channels = x.shape[3].value
                    skip_connect_channels = skip_connection.skip_value.shape[3].value
                    if default_path_channels == skip_connect_channels:
                        # If they are the same, we can simply add these two values.
                        x = keras.layers.Add()([x, skip_connection.skip_value])
                    else:
                        # Else we have to do fill the channels with Zeros or a projection:
                        if training_parameters.do_zero_padding_before_conv:
                            # Do channel ZeroPadding
                            def pad_depth(x_tmpFunc, desired_channels):
                                y = K.zeros_like(x_tmpFunc)
                                new_channels = desired_channels - x_tmpFunc.shape.as_list()[-1]
                                for _ in range(int(new_channels/x_tmpFunc.shape[3].value+1)):
                                    y = Concatenate(axis=-1)([y,y])
                                y = y[..., :new_channels]
                                return Concatenate(axis=-1)([x_tmpFunc, y])

                            # We need to pad the smaller number of channels to the bigger number of channels.
                            desired_channels = max(default_path_channels, skip_connect_channels)
                            # To implement the padding function, we need to use a Lambda Layer.
                            lambda_zeroPad = Lambda(lambda x: pad_depth(x,desired_channels))
                            # See which value has to be padded
                            if default_path_channels < skip_connect_channels:
                                # x has less channels than the skip_value. So it has to be padded.
                                tmp_x_value = lambda_zeroPad(x)
                                x = keras.layers.Add()([tmp_x_value, skip_connection.skip_value])
                            else:
                                # The skip_value has less channels than x. So the skip_value has to be padded.
                                tmp_skip_value = lambda_zeroPad(skip_connection.skip_value)
                                if tmp_skip_value.shape.dims[1] == x.shape.dims[1] and \
                                        tmp_skip_value.shape.dims[2] == x.shape.dims[2] and \
                                        tmp_skip_value.shape.dims[3] == x.shape.dims[3]:
                                    same_dimensions = True
                                else:
                                    same_dimensions = False
                                if training_parameters.do_1x1_conv_when_dim_conflict and not same_dimensions:
                                    tmp_filter_nr_skip = tmp_skip_value.shape[3].value
                                    tmp_skip_value = Conv2D(tmp_filter_nr_skip,(1,1),strides=(2,2),name="stride_skip_connection_"+str(tmp_filter_nr_skip))(tmp_skip_value)
                                x = keras.layers.Add()([x,tmp_skip_value])

                        # Comment saved for later
                        # Folgendes drin Lassen! Funktionierendes ZeroPadding:
                        # Todo:  Projection as alternative to ZeroPadding
                        # Projektion (später Auswahl implementieren)
                        # tmp_Conv_Layer = Conv2D(x.shape[3].value, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        #        kernel_initializer=tf.keras.initializers.Identity(),name="tmp_layer_"+str(number_of_run)+str(randint(1,999999999)),trainable=True)
                        # tmp_skip_value = tmp_Conv_Layer(skip_values[possibleSkip_index])


        # Build Dense Layers
        if layer.layer_type == "Dense":
            if not flattened:
                x = GlobalAveragePooling2D()(x)
                flattened = True
            x = Dense(layer.neuron_number, activation='relu', name=layer.name, trainable=old_Layers_trainable)(x)

        # Build Pooling Layer
        if layer.layer_type == "Pool":
            if not isinstance(layer, Pool_Layer):
                raise ValueError("Layer of type \"Pool\" is no object from class Pool_Layer")
            if x.shape.dims[1] == 1 or x.shape.dims[2] == 1:
                tmp_pool_size = 1
            else:
                tmp_pool_size = layer.pool_size
            tmp_stride_size = layer.pool_stride
            if layer.pooling_type == "max":
                x = keras.layers.MaxPooling2D(pool_size=tmp_pool_size, strides=tmp_stride_size, padding="valid")(x)
            elif layer.pooling_type == "average":
                x = keras.layers.AveragePooling2D(pool_size=tmp_pool_size, strides=tmp_stride_size, padding="valid")(x)
            else:
                raise ValueError("Layer of type \"Pool\" has pooling_type "+ str(layer.pooling_type))
    # End of building blocks

    # Check if already flattened for potential Dense Layers
    if not flattened:
        x = GlobalAveragePooling2D()(x)
        flattened = True

    # Output Layer
    predictions = Dense(num_classes, activation='softmax',name="SoftmaxDense",trainable=old_Layers_trainable)(x)

    # Technical Model building
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    # Import weights of earlier nets
    if(importWeights):
        model.load_weights(os.path.join(OUTPUT_DIR2, 'models', 'currModel.hdf5'), by_name=True)

    # Use MorphNet functions
    class MorphnetMetrics:
        def __init__(self, ops, regularizer_strength=morphRegulStrength, **kwargs):
            self._network_regularizer = flop_regularizer.GroupLassoFlopsRegularizer(ops, **kwargs)
            self._regularization_strength = regularizer_strength
            self._regularizer_loss = (self._network_regularizer.get_regularization_term() * self._regularization_strength)
            self._sess = K.get_session()

        def loss(self, *args):
            with self._sess.as_default():
                return keras.losses.categorical_crossentropy(*args) + self._regularizer_loss

        def flops(self, *args):
            return self._network_regularizer.get_cost()

        def regularizer_loss(self, *args):
            return self._regularizer_loss

        def calc_number_of_active_layers(self, *args):
            tmpExport = structure_exporter.StructureExporter(self._network_regularizer.op_regularizer_manager)
            with self._sess.as_default():
                number_of_active_layers = 0
                structure_exporter_tensors = self._sess.run([tmpExport.tensors])
                for layers in structure_exporter_tensors:
                    for layer in layers.values():
                        # Auf diese Weise werden alle Layer gleich gewichtet.
                        # TODO: Generate usefull weights for LayerLasso
                        number_of_channels = 0
                        for single_channel in layer:
                           if single_channel:
                            number_of_channels += 1
                        if not number_of_channels == 0:
                            number_of_active_layers += 1
                return number_of_active_layers

        # def layer_lasso_loss(self, *args):
        #     layer_regularization_strength = regularization_parameters.loss_layer_lasso_reg_strength
        #     with self._sess.as_default():
        #         return keras.losses.categorical_crossentropy(*args) + self._regularizer_loss + self.calc_number_of_active_layers() * layer_regularization_strength


        def exporter(self, export_directory,  *args):
            tmpExport = structure_exporter.StructureExporter(self._network_regularizer.op_regularizer_manager)
            with self._sess.as_default():
                structure_exporter_tensors = self._sess.run([tmpExport.tensors])
                #print(structure_exporter_tensors)
                tmpExport.populate_tensor_values(structure_exporter_tensors[0])
                tmpExport.create_file_and_save_alive_counts(export_directory, number_of_run)

    # Set up Tensorboard
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=path_to_tb_file,
                                                 histogram_freq=histograms,
                                                 write_graph=True,
                                                 write_images=False,
                                                 update_freq=1000)

    # Set up early stopping criteria
    stopping_cb = EarlyStopping(monitor=training_parameters.converged_criteria, min_delta=0.001, patience = training_parameters.number_of_non_changing_epochs)

    def lr_schedule(epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # Use MorphNet
    morphnet_metrics = MorphnetMetrics([model.output.op], threshold=morphThreshold)

    # Default value is Adam.
    model_optimizer = keras.optimizers.Adam(lr=lr_schedule(0))
    #model_optimizer = keras.optimizers.Adam()

    # Get model optimizer from config
    if training_parameters.optimizer == "SGD":
        model_optimizer = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        #model_optimizer = keras.optimizers.SGD(lr=.1, momentum=0.9, decay=0.0001)
    if training_parameters.optimizer == "rmsprob":
        model_optimizer = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

    # Compile the model
    model.compile(#loss=morphnet_metrics.layer_lasso_loss,
                  loss=morphnet_metrics.loss,
                  #optimizer=keras.optimizers.Adadelta(),
                  optimizer=model_optimizer,
                  metrics=['accuracy', morphnet_metrics.flops, morphnet_metrics.regularizer_loss])


    # Activate the used callback functions
    active_cb=[]
    if generate_tb_files:
        active_cb.append(tensorboard_cb)
    if early_stopping:
        active_cb.append(stopping_cb)

    active_cb.append(lr_scheduler)
    active_cb.append(lr_reducer)

    def run_the_model(do_data_aug):
        if do_data_aug:
            tmp_history = model.fit_generator(tmp_iterator,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_val, y_val),
                                    callbacks=active_cb)
        else:
            tmp_history = model.fit(x_train, y_train,
                          batch_size=dataset.batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_val, y_val),
                          callbacks=active_cb
                          )
        return tmp_history

    # Run the model
    r = run_the_model(do_data_augmentation)

    # Export MoprhNet structure
    if(export_morphNet_structure):
        morphnet_metrics.exporter(os.path.join(OUTPUT_DIR2,'Netze'))

    end_timer = time.time()
    print("Time:\n")
    print(end_timer - start_timer)

    if train_without_reg:
        export_weights = False
    else:
        export_weights = True

    # Export Weights
    if export_weights:
        saved_model_path=os.path.join(OUTPUT_DIR2, 'models')
        os.makedirs(saved_model_path, exist_ok=True)
        model.save_weights(os.path.join(saved_model_path, 'currModel.hdf5'))

    if plot_options.plot_architecture_vis_keras:
        visualkeras.layered_view(model, draw_volume=False, to_file='/home/burghoff/Daten/210325_visualizeOutput/output3.png').show() # write and show
        visualkeras.graph_view(model, draw_volume=False, to_file='/home/burghoff/Daten/210325_visualizeOutput/output_graph3.png').show()  # write and show

    path_to_summed_layer_weights = os.path.join(OUTPUT_DIR2, 'Netze', 'layer_weights_summed_' + str(number_of_run) + ".csv")
    with open(path_to_summed_layer_weights, 'a') as layer_weights_file:
        for layer in model.layers:
            if "New_Conv2D" in layer.name:
                weights = layer.get_weights()  # list of numpy arrays
                kernel_weights = weights[0]
                absolute_sum = sum(sum(sum(sum(abs(kernel_weights)))))
                number_of_weights = kernel_weights.size
                val_per_weight = float(absolute_sum) / float(number_of_weights)
                layer_weights_file.write(layer.name + "," + str(val_per_weight) + "\n")

    all_weights_output = True
    path_to_summed_layer_weights2 = os.path.join(OUTPUT_DIR2, 'Netze',
                                                'all_layer_weights_summed_' + str(number_of_run) + ".csv")
    if all_weights_output:
        with open(path_to_summed_layer_weights2, 'a') as layer_weights_file:
            for layer in model.layers:
                if "Conv2D" in layer.name:
                    weights = layer.get_weights()  # list of numpy arrays
                    kernel_weights = weights[0]
                    absolute_sum = sum(sum(sum(sum(abs(kernel_weights)))))
                    number_of_weights = kernel_weights.size
                    val_per_weight = float(absolute_sum) / float(number_of_weights)
                    layer_weights_file.write(layer.name + "," + str(val_per_weight) + "\n")

    if not dummy_training:
        # flopcosts after each training
        flopcost_file = os.path.join(OUTPUT_DIR2,'Netze', 'learned_structure', 'flopcosts.csv')
        with open(flopcost_file, 'a') as flop_file:
            flop_file.write(str(number_of_run) + "," + str(r.history["flops"][len(r.history['flops'])-1]) + "\n" )

    # flopcosts before MorphNet shrinking + expanding
    flopcost_file2 = os.path.join(OUTPUT_DIR2, 'Netze', 'learned_structure', 'morph_flopcosts.csv')
    with open(flopcost_file2, 'a') as flop_file2:
        flop_file2.write(str(number_of_run) + "," + str(r.history["flops"][len(r.history['flops']) - 1]) + "\n")


    # Print results
    score = model.evaluate(x_test, y_test, verbose=0)
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    if default_acc_output:
        path_to_acc_csv = os.path.join(OUTPUT_DIR1, 'test_accuracy.csv')
        path_to_loss_csv = os.path.join(OUTPUT_DIR1, 'test_loss.csv')
        path_to_train_acc_csv = os.path.join(OUTPUT_DIR1, 'train_accuracy.csv')
        path_to_train_loss_csv = os.path.join(OUTPUT_DIR1, 'train_loss.csv')
    else:
        if train_without_reg:
            path_to_acc_csv = os.path.join(OUTPUT_DIR1, 'no_reg_test_accuracy.csv')
            path_to_loss_csv = os.path.join(OUTPUT_DIR1, 'no_reg_test_loss.csv')
            path_to_train_acc_csv = os.path.join(OUTPUT_DIR1, 'no_reg_train_accuracy.csv')
            path_to_train_loss_csv = os.path.join(OUTPUT_DIR1, 'no_reg_train_loss.csv')
        else:
            raise ValueError("No directory specified where to put the accuracy output.")

    def gen_new_file_if_not_exist(my_tmp_path, attr):
        if not os.path.exists(my_tmp_path):
            with open(my_tmp_path, 'a') as file:
                file.write('Current_run,' + str(attr) + "\n")

    gen_new_file_if_not_exist(path_to_acc_csv, "Test_acc")
    gen_new_file_if_not_exist(path_to_loss_csv, "Test_loss")
    gen_new_file_if_not_exist(path_to_train_acc_csv, "Train_acc")
    gen_new_file_if_not_exist(path_to_train_loss_csv, "Train_loss")

    def add_new_data_to_file(path_to_adding_file, data_array):
        with open(path_to_adding_file, 'a') as file:
            for tmp_element in data_array:
                file.write(str(tmp_element) + ",")
            file.write("\n")

    if not dummy_training:
        add_new_data_to_file(path_to_acc_csv, [number_of_run, score[1]])
        add_new_data_to_file(path_to_loss_csv, [number_of_run, score[0]])
        add_new_data_to_file(path_to_train_loss_csv, [number_of_run, train_score[0]])
        add_new_data_to_file(path_to_train_acc_csv, [number_of_run, train_score[1]])

        path_to_write_no_tr_it = os.path.join(OUTPUT_DIR1, 'number_of_epochs_before_converged.csv')
        if not os.path.exists(path_to_write_no_tr_it):
            with open(path_to_write_no_tr_it, 'a') as file:
                file.write("Number_of_run,Epochs_trained\n")
        with open(path_to_write_no_tr_it, 'a') as file:
            trained_epochs = len(r.history['loss'])
            file.write(str(number_of_run) + "," + str(trained_epochs)+"\n")

