import os
from shutil import copyfile
import config

from config import environment_parameters

# new_path = callFunction.Default_directory_total

#path = "/home/burghoff/Daten/201005_test_03"
def copy_files_and_print_console_commands():
    path_of_project_folder = environment_parameters.project_folder
    path_of_new_script = environment_parameters.working_directory

    if not os.path.exists(path_of_new_script):
        os.makedirs(path_of_new_script)
        os.makedirs(os.path.join(path_of_new_script, "customFunctions"))
        # "directoryHelper.py" and "training_helper" are not copied at the moment because they are not used.
        files_to_copy_in_custom_funcions = ["autoaugment.py", "changeArchitecture.py", "environmentFunctions.py", "pipelineFunctions.py", "labels.py"]
        files_to_copy = ["config.py", "main.py", "pipeline.py", "plots.py", "training.py"]
        for file in files_to_copy:
            copyfile(os.path.join(path_of_project_folder, file), os.path.join(path_of_new_script, file))
        for file in files_to_copy_in_custom_funcions:
            copyfile(os.path.join(path_of_project_folder, "customFunctions", file), os.path.join(path_of_new_script, "customFunctions", file))

    datasets = environment_parameters.all_possible_datasets
    for dataset in datasets:
        if config.often_used_variables.test_environment:
            print("WARNUNG: Test environment active!")
        print("nohup python3 " + str(path_of_new_script) + "/main.py \"runInConsole\" \"" + str(dataset) + "\" > "
              + str(path_of_new_script) + "/run_" + str(dataset) + ".out &")
    #print("nohup python3 " + str(path_of_new_script) + "/main.py \"runInConsole\" > " + str(
    #    path_of_new_script) + "/run.out &")