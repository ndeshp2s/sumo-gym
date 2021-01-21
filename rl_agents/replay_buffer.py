import random
import numpy as np
import torch
from collections import deque, namedtuple
import pickle

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

    def __getstate__(self):
        print("I'm being pickled")
        return self.memory

    def __setstate__(self, d):
        print("I'm being unpickled with these values: " + repr(d))
        self.memory = d

    # def save_buffer(self):

    #     with open('replay_memory_buffer.pkl', 'wb') as fp:
    #         pickle.dump(self.memory, fp)
    #     # with bz2.BZ2File(self.buffer_dir + '/replay_memory_buffer.bz2', 'wb') as f:
    #     #     pickle.dump(self.agent.buffer.memory, f)
