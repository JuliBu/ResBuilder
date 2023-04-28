import os
from datetime import date

class often_used_variables:
    """
    Here is the collection of variables which are often used.
    """
    used_net = "emptyNet"
    startnet_name = used_net + ".json"
    short_run_description = "highFlops_lowEpochs20_" + str(used_net) + "_v01"
    test_environment = False
    datasets = ["MNIST"] # Which dataset should be used when main is executed

class environment_parameters:
    """
    Here is a bunch of parameters regarding GPU, directories, ... which have to be set up in order to use the code
    """

    ### Set these parameters manually: ###

    default_gpu_index = 0
    """In case of a multi-GPU system you can set which GPU is used for computation."""

    # how much percentage of your gpu-RAM can a single process allocate (miniImageNet takes the double of gpu_fragment_default)
    gpu_fragment_default = 0.99
    """You can set how much percentage of your GPU-RAM is allocated at maximum."""

    allow_gpu_growth = True
    """Sets if the Process can grow on VRAM on the gpu. If set false it allocates directly the full fragment."""

    default_parent_directory = os.path.join(os.path.expanduser('~'), 'Daten')
    """Sets the path where the experiments and other folders are saved"""

    short_run_description = often_used_variables.short_run_description
    """A short desciption of the run which is used for the outputfolder name."""

    test_environment = often_used_variables.test_environment
    """Set this variable to True if you want to set the number of epochs to 1 and ........ #ToDo """

    debugging = False
    """This variable is used in developement to activate some Beakpoints."""

    project_folder = "/home/burghoff/VSCodeProjects/git_autoMLWithMorphNet/automlwithmorphnet"
    """Directory where the PyCharm project is located on the remote machine."""

    #startNet_folder = "/home/burghoff/Daten/200731_startnetze/"
    startNet_folder = "/home/burghoff/Daten/220502_startnets_new_format/"
    """Folder in which the startnets are kept."""

    #startNet_name = "211201_emptyREsNet_largeWithDense.json"
    startNet_name = often_used_variables.startnet_name
    """Name of the startnet."""

    # Directory where datasets that are not available in tensorflow datasets are saved:
    custom_dataset_directory = "/home/burghoff/kleineDatensaetze/"

    ### These following parameters are set automatically ###

    today_date = date.today().strftime("%y%m%d")
    """Function which gives the current Date in form yymmdd"""

    # Current Output Path
    output_folder_name = today_date + short_run_description
    """Sets the folder where the run is saved. For full path see "working_directory"."""

    working_directory = os.path.join(default_parent_directory, output_folder_name)
    """This is the path to the folder where the run is saved."""

    all_possible_datasets = ["CIFAR10", "FashionMNIST", "MNIST", "miniImageNet", "smallNORB", "EMNIST", "CIFARhundred", "GTSRB", "TinyImageNet", "Animals10"]
    """Here is a list of all possible datasets which is used for things like printing console commands or plotting"""

class pipeline_parameters:
    """
    Here can parameters like the number of morphnet Iterations and number of layer-adding-steps be set
    """

    number_morph_steps = 7 #7
    """Sets, how many MorphNet steps should be done in training."""
    number_layer_steps = 3 #3
    """Sets, how often a layer is inserted. (How many layer are inserted before the next MorphNet-step in case of "outerMorphNet" as pipeline architecture)."""

    type_of_adding_layers = "one_block"
    """Sets which type of layer(blocks) shpuld be added by an insertion step. Possible values are: "one_block". """
    number_of_layers_added = 2
    """Sets which design of layer(block) is inserted in every insertion step. 
    Possbile values are: "1" for 1 layer with a skip around, 2 for 2 layers with a skip around, ..."""

    position_of_adding = "equal_blockwise"
    """Sets where the new layer(block) is added at an insertion step. Possible values are: "random", "after_first", "after_ending_skip", "equal_blockwise". """

    avoid_double_architectures = True
    """Sets if by every adding of layers it should be checked if the architecture after adding has been trained on before."""
    attemps_to_get_new_architecture = 10
    """If avoid_double_architectures: Sets how often it is tried to generate a new """

    use_layer_lasso_momentum = True
    """Sets if the deletion step is skipped if there was an accuracy boost in the last adding"""
    layer_lasso_momentum_threshold = 0.015
    """How high has the accuracy boost to be in order to skip the deletion step"""
    delete_after_shrinkage_step = True
    """If use_layer_lasso_momentum is true, this sets if the skip should also be done after the shrinkage step"""
    waiting_time_before_delete = 45
    """Sets the time in seconds before the deletions step starts, in order to have all writing processes finished before reading access."""

    delete_layers_with_0_channels = True

class training_parameters:
    """
    Collection of parameters which are important for each training iteration. For example: When stop training? Data Augmentation? Much more Hyperparameters.
    """

    train_till_converged = False
    """Set to True, if you want to end the training after the "converged_criteria" did not change for "number_of_non_changing_epochs"."""

    number_of_non_changing_epochs = 25   # Unneccessary if train_till_converged==False
    """If train_till_converged this is the number of epochs, the converged_criteria must not have improved in order to end training."""
    converged_criteria = 'loss'      # Unneccessary if train_till_converged==False
    """If train_till_converged this criteria must not have improved for number_of_non_changing_epochs epochs in order to end training."""
    exact_number_of_epochs = 20         # Unneccessary if train_till_converged==True
    """If not train_till_converged this is the number of epochs per training."""
    #ToDo
    factor_number_of_epochs_in_layerlasso = 0.25
    factor_learning_rate = 0.1

    # Should data augmentation be used?
    do_training_with_data_augmentation = True
    """Sets if Data Augmentation should be done."""
    use_complex_data_aug = True
    """There are two types of data augmentation to choose from: simple and complex where complex does improve the results in trade for performance."""

    # Shall the data be preprocessed
    do_data_preprocessing = True
    """Does preprocessing (Data-Mean)/Std. Mean and std are calculated on train dataset if not otherwise specified"""

    # Should Batch Normalition be used?
    do_batch_normalization = True
    """Decides whether BN is used."""

    ignore_strides_in_convs = False
    """Sets if the strides of conv layers are ignored. Useful for datasets with small pixel dimensions like CIFAR10."""

    do_1x1_conv_when_dim_conflict = True
    """Adds a 1x1 conv layer on the skip connections if dimensions of feature maps do not fit (see e.g. ResNet18)."""

    when_train_without_reg = 1
    """Every how many trainings should be training without regularization done?"""

    do_zero_padding_before_conv = True
    """If True a zero-padding in the channel size is done, so that there are no problems with dimensions."""

    optimizer = "Adam"
    """Which optimizer should be used? Possbile Optimizers are: "Adam", "SGD", "rmsprob". Best results so far have been with "Adam"."""

    # If we only test (for example for debugging) the pipeline, we only train 1 epoch each training.
    if environment_parameters.test_environment:
        train_till_converged = False
        exact_number_of_epochs = 1

class regularization_parameters:
    """
    Here you can find all parameters for the MorphNet and LayerLasso Regularization
    """
    # MorphNet Parameters:
    morphNet_Lambda = 1e-7# 1e-7
    """The Lambda for MorphNet regularozation which influences the Lossfunction."""
    morph_intensity_shrinkage = 1
    """Morph intensity: 1 for default morph_step shrinkage, 0.x for x Percent shrinkage of Morphing Step"""
    morph_intensity_expanding = 0.5
    """Morph intensity: 1 for default morph_step expanding, 0.x for x Percent expanding of Morphing Step"""
    target_flops = 1000000000 #low:10000000 middle:100000000 default: 1000000000
    """Target Flops for a net in each expanding MorphNet step"""

    # LayerLasso Parameters:
    layer_lasso_reg_strength = 1e-8 # 1e-10
    """Regularization strength of new Layers."""
    #layer_deleting_threshold = 1e-03
    layer_deleting_threshold = 1e-03
    """Under which threshold a layer should be kicked."""
    #loss_layer_lasso_reg_strength = 0*1e-4 # not used at the moment
    #"""LayerLassoLoss should not effect the outcome."""

    # Additionnal regularization
    do_additional_l2_reg = True
    """Sets if l2_regularization is done."""
    l2_regularization = 1e-5 #1e-5
    """Sets l2_regularization_strength"""

class input_parameters:
    """
    Here you can set input parameters like the used dataset
    """

    datasets = often_used_variables.datasets
    """Dataset can be ["MNIST", "FashionMNIST", "CIFAR10", "miniImageNet", "smallNORB", "EMNIST"]. It might be overwritten by the runRoutine call."""

    split_a_seperate_val_set = True
    percentage_val_from_train = 0.2

class output_parameters:
    """
    Here are the parameters regarding outputs like tensorboard
    """
    # Tensorboard Parameters
    tensorboard_output = True
    """Should tensorboard files be generated?"""
    tensorboard_hists = 0
    """Log the tensorboard histograms? Much more computational time."""
    show_images_before_aug = False
    """Saves a figure with some images of the dataset before the data augmentation takes place"""
    show_images_after_aug = False
    """Saves a figure with some images of the dataset after the data augmentation took place"""
    image_indices = [10,810,2510,9510] #[1,2,3,4]
    """Sets which images of the dataset are shown"""

class plot_options:
    """
    Here you can set what should be plotted while training
    """
    plot_no_reg_points = True
    """Plot the Accuracies of the no_reg_runs."""
    plot_while_training = True
    """Plot while training?"""

    plot_architecture_vis_keras = False
    """Beta: Does not work! (Plot the architecture while training.)"""

    generate_acc_plots_while_training = True
    """Sets if a plot after each morphNet iteration should be given"""
    if pipeline_parameters.number_layer_steps == 0:
        generate_acc_plots_while_training = False

    plot_arch_HarisIqbal = True
    path_to_layer_folder = "/home/burghoff/Nebenpakete/PlotNeuralNet/"
    image_scale_factor = 1
    orig_image_size = (32,32)
    width_shrinkage_scale = 0.5
    #ToDo! get image size from current dataset!



