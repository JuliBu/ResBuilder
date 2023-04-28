"""
This file contains the architecture of the pipeline.
"""
# Import Libraries
import os
# Import own scripts
import plots
from config import environment_parameters, input_parameters, pipeline_parameters, training_parameters, \
    regularization_parameters, plot_options
import customFunctions.pipelineFunctions as pFunctions
from customFunctions import changeArchitecture
from customFunctions.changeArchitecture import layer_delete_routine, layer_adding_routine, morphnet_call_function


def trainings_pipeline_outer_morphNet(only_these_datasets = input_parameters.datasets):
    """
        This is the main pipeline. In every step we add a layer(block) and every "number_of_adding_layers" step, there is a MorphNet step. At all there are "morphNet_Iterations" morphNet steps done.
        So all in all there are "number_of_adding_layers" times "morphNet_Iterations" many steps to train. It starts many runs of training.
        :param morphNet_Iterations: How many MorphNet steps should be done in total
        :param number_of_adding_layers: How often should a layer(block) be added before a morphNet step takes place.
        :param default_dir: This is the working directory, where all the data and results are saved.
        :param only_these_datasets: Sets for which datasets the pipeline is tested. Default value is in the config file. But in most cases it is overwritten by the main function.
        :return: None
    """
    morphNet_Iterations = pipeline_parameters.number_morph_steps
    number_of_adding_layers = pipeline_parameters.number_layer_steps
    default_dir = environment_parameters.working_directory


    datasets = pFunctions.get_datasets(only_these_datasets=only_these_datasets)

    def run_trainings(dataset, curr_run, path_to_net, from_scratch=False):
        # Train the new net with regularization
        if from_scratch:
            pFunctions.train_from_scratch(dataset, curr_run, path_to_net)
        else:
            pFunctions.train_further(dataset, curr_run, path_to_net)

        # Train the net without regularization
        if curr_run % training_parameters.when_train_without_reg == 0:
            pFunctions.train_without_reg(dataset, curr_run, path_to_net, import_weights=True)
            pFunctions.train_without_reg(dataset, curr_run, path_to_net, import_weights=False)

    for dataset in datasets:
        Default_directory = os.path.join(default_dir, dataset)
        path_to_net = os.path.join(Default_directory, 'Netze', 'startnet.json')
        logfile_path = os.path.join(Default_directory, 'logfile.txt')
        curr_run = 0
        used_nets = []
        for new_from_scratch in range(morphNet_Iterations):
            #pFunctions.train_from_scratch(dataset, curr_run, path_to_net)
            for curr_run_tmp in range(number_of_adding_layers+1):
                curr_run = curr_run+1
                # In the first run after training from scratch just train the net
                if curr_run_tmp == 0:
                    run_trainings(dataset, curr_run, path_to_net, from_scratch=True)
                    if not curr_run == 1:
                        # Deleting Layers
                        path_to_net = layer_delete_routine(Default_directory, curr_run, path_to_net,
                                                           regularization_parameters.layer_deleting_threshold,
                                                           delete_before_training=True)
                        used_nets.append(changeArchitecture.gen_net_from_json_path(path_to_net))
                else:
                    # Adding a structure of layers
                    path_to_net = layer_adding_routine(Default_directory, curr_run, path_to_net, logfile_path, used_nets=used_nets)
                    used_nets.append(changeArchitecture.gen_net_from_json_path(path_to_net))
                    # Run new trainings
                    run_trainings(dataset, curr_run, path_to_net)
                    # Deleting Layers
                    path_to_net = layer_delete_routine(Default_directory, curr_run, path_to_net,
                                                       regularization_parameters.layer_deleting_threshold,
                                                       delete_before_training=True)
                    used_nets.append(changeArchitecture.gen_net_from_json_path(path_to_net))

            if not training_parameters.when_train_without_reg == 1:
                pFunctions.train_without_reg(dataset, curr_run, path_to_net)

            # Begin MorphStep
            pFunctions.train_dummy(dataset, curr_run, path_to_net)

            path_to_net = morphnet_call_function(Default_directory, curr_run, path_to_net)
            used_nets.append(changeArchitecture.gen_net_from_json_path(path_to_net))
            if plot_options.generate_acc_plots_while_training:
                plots.plot_layer_and_morphing_steps(default_dir, only_dataset=dataset)
