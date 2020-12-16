import random
import numpy as np
import torch
from collections import deque, namedtuple


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, state, action, reward, next_state, done):
        experience = self.experience(state = state, action = action, reward = reward, next_state = next_state, done = done)
        self.memory.append(experience)


    def sample(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size

        experiences = random.sample(self.memory, k = batch_size)

        return experiences

    def __len__(self):
        return len(self.memory)