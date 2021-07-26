
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable


class PrefNet(nn.Module):

    def __init__(self, nb_obj=2, middle_size=10, **kwargs):
        super().__init__()
        self.f1 = nn.Linear(nb_obj, middle_size)
        self.f2 = nn.Linear(middle_size, middle_size)
        self.out = nn.Linear(middle_size, 1)

    def forward(self, X):
        y = F.relu(self.f1(X))
        y = F.relu(self.f2(y))
        return self.out(y)

    def predict(self, X):
        X_torch = torch.from_numpy(X).float()
        with torch.no_grad():
            return self.forward(X_torch).data.numpy()

    def loading(self, path, **kwargs):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return self
