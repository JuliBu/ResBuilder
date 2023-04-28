





# not used at the moment





class Training_Step:
    def __init__(self, current_net: str):
        self.current_net = current_net

class Full_Training:
    def __init__(self, dataset: str, type="outer_morphnet"):
        self.dataset = dataset
        self.type = type
