"""
This is the script which is executed to start the trainings pipeline
"""
import pipeline
import sys
import customFunctions.environmentFunctions



# If you want to run the pipeline start the main function with this parameter.
# All hyperparameters are loaded from the config file.
if sys.argv[1] == "runOuterMorphNet":
    pipeline.trainings_pipeline_outer_morphNet()
# You can also run the copy and print console commands mode, which makes it easier to run this code in console.
elif sys.argv[1] == "printCommands":
    customFunctions.environmentFunctions.copy_files_and_print_console_commands()
elif sys.argv[1] == "runInConsole":
    dataset = []
    dataset.append(sys.argv[2])
    pipeline.trainings_pipeline_outer_morphNet(only_these_datasets=dataset)
else:
    raise ValueError("Invalid argument for main function!")