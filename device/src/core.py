# device/src/core.py
import numpy as np
from src.veectordb import VeectorDB  # Correct path import

class Veector:
    def __init__(self, db_path="/workspaces/Veector/device/data/db/user_data.json"):  # Correct Path to
        self.db = VeectorDB(db_path)
        self.space = {} # Simple in-memory space
        self.max_coord = 0 # Simple coordinate tracking

    def _next_coords(self):
        coords = max([key[1][0] for key in self.space.keys()] + [self.max_coord]) + 1
        self.max_coord = coords
        return [coords, coords, coords]

    def add_to_space(self, tensor):
        layer, coords = tensor[0][0], tuple(tensor[0][1])
        self.space[(tuple(layer), coords)] = tensor # Store tensor directly

    def compute(self, tensor):
        # Placeholder compute function
        print(f"Computing tensor: {tensor}")
        return np.random.rand(1, 512)  # Dummy output
