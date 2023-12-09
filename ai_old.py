import random
import math
from game import Game

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

N_OBSERVATIONS = 6
N_ACTIONS = 1


class Memory:
    def __init__(self, capacity=10000):
        self.data = np.empty((capacity), dtype=object)
        self.position = 0

    def push(self, state, action, next_state, reward):
        self.data[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.data.shape[0]

    def sample(self, batch_size):
        return np.random.choice(self.data, batch_size)

    def __len__(self):
        return len(self.data)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.input = nn.Linear(N_OBSERVATIONS, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.output = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


cur_step = 0
model = DQN()
optimizer = optim.Adam(model.parameters())
memory = Memory()

EPS_START = 0.9
EPS_END = 0.05
EPS_STEPS = 200
BATCH_SIZE = 128


def get_action(state):
    global cur_step
    epsilon = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * cur_step / EPS_STEPS)
    cur_step += 1
    if epsilon > threshold:
        with torch.no_grad():
            return model(torch.tensor(state).float()).argmax().item()
    else:
        return random.randint(0, 1)


def train():
    if len(memory) < BATCH_SIZE:
        return
    batch = memory.sample(BATCH_SIZE)
    batch = np.array(batch).transpose()  # Really needs to transpformed to array?
    states = np.vstack(batch[0])
    actions = list(batch[1])
    next_states = np.vstack(batch[2])
    rewards = list(batch[3])
    actions = (
        model(torch.tensor(states).float())
        .gather(1, torch.tensor(actions).unsqueeze(-1))
        .squeeze(-1)
    )


def data_func(data: np.ndarray):
    pass


game = Game()
game.run(data_func=data_func)
