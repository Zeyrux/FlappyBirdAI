import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name: str = "model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)


class Trainer:
    def __init__(self, model: DQN, lr: float, gamma: float):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)
        action = torch.Tensor(action)
        reward = torch.Tensor([reward] if type(reward) == float else reward)

        if len(state.shape) == 1:
            # if the state is not batched
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        prediction = self.model(state)

        # 2: r + y * max(next_predicted Q value)
        target = prediction.clone()
        for i in range(len(done)):
            q_new = reward[i]
            if not done:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
