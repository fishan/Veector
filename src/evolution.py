from core import Veector

class Evolution:
    def __init__(self, veector):
        self.veector = veector

    def evolve(self, tensor):
        return self.veector.compute([
            tensor[0],
            [[0], tensor[0][1], [9, 0, 0], 1],  # Reason
            tensor[2],
            tensor[3],
            []
        ])