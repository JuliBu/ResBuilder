"""
Here are helper functions which are needed to build the pipeline
"""
# import homemade functions
import training
from config import training_parameters, output_parameters


def get_datasets(only_these_datasets):
    """
    Sets a list of default datasets if no dataset is set.
    :param only_these_datasets: List of datasets
    :return: List of datasets
    """
    if only_these_datasets == "None":
        datasets = ["MNIST", "FashionMNIST", "CIFAR10"]
    else:
        datasets = []
        for dataset in only_these_datasets:
            datasets.append(dataset)
    return datasets


# Training methods
def run_any_training(my_dataset, my_run, my_path_to_net,
                 import_weights=True,
                 dummy_training=False,
                 train_without_reg=False,
                 export_morphNet_structure=True,
                 number_of_epochs=training_parameters.exact_number_of_epochs,
                 tensorboard_output=output_parameters.tensorboard_output,
                 tensorboard_hists=output_parameters.tensorboard_hists,
                 train_till_converged=training_parameters.train_till_converged):
    """
    Default training method which is called by more specific training methods.
    """
    training.run_training(my_run, number_of_epochs,
                          generate_tb_files=tensorboard_output, histograms=tensorboard_hists,
                          export_morphNet_structure=export_morphNet_structure, importWeights=import_weights,
                          early_stopping=train_till_converged,
                          name_of_data=my_dataset, path_to_net=my_path_to_net,
                          old_Layers_trainable=not dummy_training, dummy_training=dummy_training,
                          train_without_reg=train_without_reg)


def train_from_scratch(my_dataset, my_run, my_path_to_net):
    """
    Trains a net from scratch. So no Weight import is done.
    """
    run_any_training(my_dataset, my_run, my_path_to_net, import_weights=False)


def train_further(my_dataset, my_run, my_path_to_net):
    """
    Imports weights from former training.
    """
    run_any_training(my_dataset, my_run, my_path_to_net)


def train_without_reg(my_dataset, my_run, my_path_to_net, import_weights=False):
    """
    Calls the run_training without regularization and with no import weights.
    The run-Counter is set to +10000 (no imported weights) or +20000 (imported weights) so that you can see later,
    which runs were done without regularization because these runs are normally no part of the trainingspipeline.
    Weights are not exported after this layer_type of training (See "Export Weights" in run_training).
    """
    if import_weights:
        added_counter = 20000
    else:
        added_counter = 10000

    run_any_training(my_dataset, my_run + added_counter, my_path_to_net, import_weights=import_weights,
                     train_without_reg=True, export_morphNet_structure=False)


def train_dummy(my_dataset, my_run, my_path_to_net):
    """
    Only run one epoch of "training" (all layers frozen) to print some information about the current architecture
    """
    run_any_training(my_dataset, my_run, my_path_to_net, dummy_training=True, export_morphNet_structure=False,
                     number_of_epochs=1, tensorboard_output=False, tensorboard_hists=0, train_till_converged=False)


