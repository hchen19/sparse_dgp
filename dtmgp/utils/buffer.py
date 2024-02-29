import numpy as np
import torch
import torch.nn as nn


class TmkBuffer:
    def __init__(self,
                 capacity,
                 feature_dim,
    ):
        self.capacity = capacity
        self.feature_dim = feature_dim

        self.invs = np.empty((capacity, feature_dim, feature_dim), dtype=np.float32)

        self.count = 0

    def store(self, inv):
        np.copyto(self.invs[self.count], inv)
        self.count += 1

    def numpy_to_tensor(self):
        inv = torch.tensor(self.invs, dtype=torch.float32)
        return inv