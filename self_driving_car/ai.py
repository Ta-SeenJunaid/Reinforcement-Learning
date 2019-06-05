
#importing libraries

import numpy as np
import random
import os
import tourch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tourch.autograd as autograd
from tourch.autograd import Variable

#architecture of nural network

class Network(nn.Module):
    
    def __init__(self, input_size,nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.full_connection_1 = nn.Linear(input_size, 30)
        self.full_connection_2 = nn.Linear(30, nb_action)

