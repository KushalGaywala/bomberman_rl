import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    input_size = 17 * 17 * 6
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up DQN model from scratch.")
        self.model = DQN(input_size=input_size, output_size=len(ACTIONS))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    else:
        self.logger.info("Loading DQN model from saved state.")
        self.model = DQN(input_size=input_size, output_size=len(ACTIONS))
        self.model.load_state_dict(torch.load("my-saved-model.pt"))

    self.epsilon = 0.1


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train and random.random() < self.epsilon:
        return np.random.choice(ACTIONS)
    
    state = state_to_features(game_state)
    q_values = self.model(torch.FloatTensor(state))
    return ACTIONS[torch.argmax(q_values).item()]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    arena = game_state['field']
    agent_pos = game_state['self'][3]
    bombs = game_state['bombs']
    coins = game_state['coins']
    enemies = [enemy[3] for enemy in game_state['others']]
    explosion_map = game_state.get('explosion_map', np.zeros_like(arena))

    channels = [
        arena,
        explosion_map,
        np.array([[1 if (x, y) == agent_pos else 0 for x in range(arena.shape[1])] for y in range(arena.shape[0])]),
        np.array([[1 if (x, y) in coins else 0 for x in range(arena.shape[1])] for y in range(arena.shape[0])]),
        np.array([[1 if (x, y) in enemies else 0 for x in range(arena.shape[1])] for y in range(arena.shape[0])]),
        np.array([[1 if any((x, y) == bomb[0] for bomb in bombs) else 0 for x in range(arena.shape[1])] for y in range(arena.shape[0])])
    ]

    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)
