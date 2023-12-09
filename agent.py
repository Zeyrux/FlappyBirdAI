from collections import deque
from random import sample, randint

from game import Game
from model import DQN, Trainer

import torch
import numpy as np

# https://www.youtube.com/watch?v=PJl4iabBEz0


MAX_MEMORY = 100_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class Agent:
    def __init__(self, game: Game) -> None:
        self.n_games = 0
        self.eplison = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.game = game
        self.model = DQN(2, 8, 2)
        self.trainer = Trainer(self.model, LEARNING_RATE, self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        n_games_exploration = 500
        self.eplison = n_games_exploration - self.n_games
        action = [0, 0]
        if randint(0, n_games_exploration // 2) < self.eplison:
            move = randint(0, 1)
            action[move] = 1
        else:
            state = torch.Tensor(state)
            prediction = self.model(state)
            move = torch.argmax(prediction).item()
            action[move] = 1
        return action


def train():
    game = Game()
    agent = Agent(game)
    game.start()
    rewards = []
    while True:
        state_old = game.get_state()
        action = agent.get_action(state_old)
        reward, done, score = game.step(action)
        rewards.append(reward)
        state_new = game.get_state()

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print(f"Game {agent.n_games} Score: {np.array(rewards).sum()}")
            rewards = []
            if agent.n_games % 1 == 0:
                agent.model.save(file_name=f"model.pth")
                pass


if __name__ == "__main__":
    train()
