
# not used at the moment


from config import environment_parameters as eP
import os


class Directory:
    def __init__(self, upper_dir: str):
        self.upper_dir = upper_dir

class Dataset_experiment_dir(Directory):
    def __init__(self, upper_dir: str, dataset: str):
        super().__init__(upper_dir=upper_dir)
        self.dataset = dataset

    def dataset_experiment(self, dataset: str) -> str:
        return os.path.join(self.upper_dir, dataset)

def dataset_experiment_direcotry(dataset: str) -> str:
    return os.path.join(eP.working_directory, dataset)

def path_to_test_acc(dataset: str) -> str:
    os.path.join(eP.working_directory, str(dataset), "test_accuracy.csv")

