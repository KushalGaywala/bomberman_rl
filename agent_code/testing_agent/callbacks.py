import os
import pickle
import random
import numpy as np

# Possible actions in the game
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EXPLORATION_RATE = 0.1  # Exploration rate, defined here to maintain consistency

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    if self.train or not os.path.isfile("winston_zeddmore.pt"):
        self.logger.info("Setting up model from scratch.")
        # Adjusting the Q-table dimensions to match the new feature space
        self.q_table = np.zeros((17, 17, 17, 17, 100, len(ACTIONS)))  # Adjusted dimensions based on features
    else:
        self.logger.info("Loading model from saved state.")
        with open("winston_zeddmore.pt", "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Choose the action to take based on the current game state.
    """
    random_prob = EXPLORATION_RATE  # 10% of the time, take a random action for exploration
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(self, game_state)
    
    # Extract the relevant part of the Q-table using the state features
    agent_x, agent_y, coin_x, coin_y, coin_dist = features[:5]
    q_values = self.q_table[int(agent_x), int(agent_y), int(coin_x), int(coin_y), int(coin_dist)]

    # Get the best action based on the Q-table
    action_idx = np.argmax(q_values)
    return ACTIONS[action_idx]

def state_to_features(self, game_state: dict) -> np.array:
    """
    Convert the game state to a feature vector.
    """
    if game_state is None:
        return None

    action = np.zeros((1, 6))

    # Extract features like the agent's position, the positions of walls, coins, and bombs
    field = np.array(game_state['field'])
    agent = np.array(game_state['self'][3])  # Agent's position
    coins = np.array(game_state['coins'])

    nearest_coin, nearest_coin_dist = get_nearest_coin(self, agent, coins)
    action = get_optimal_action(self, agent, nearest_coin, field)

    features = np.concatenate([
        agent, 
        nearest_coin, 
        np.array([nearest_coin_dist]),
        action
    ])

    self.logger.debug(f"\nFeatures:\n{features}")

    return features

def get_nearest_coin(self, agent: np.array, coins: np.array) -> np.array:
    if coins.size <= 0:
        return np.array([0, 0]), 0

    distances = coins - agent
    # sum_abs_dists = np.sum(np.abs(distances), axis=1)
    nearest_coin_dist = np.min(np.sum(np.abs(distances), axis=1))
    nearest_coin = coins[np.argmin(np.sum(np.abs(distances), axis=1))]
    return nearest_coin, nearest_coin_dist

def get_optimal_action(self, agent: np.array, coin: np.array, field: np.array) -> np.array:
    """
    Determine the optimal action based on the agent's position relative to the nearest coin.
    Returns a one-hot encoded action array.
    """

    action = np.zeros(6)

    col_diff = coin[0] - agent[0]
    row_diff = coin[1] - agent[1]

    if coin[0] == 0 and coin[1] == 0:
        action = np.array([0, 0, 0, 0, 0, 0])
    elif col_diff > row_diff and row_diff < 0:
        action = np.array([1, 0, 0, 0, 0, 0])
    elif col_diff < row_diff and col_diff > 0:
        action = np.array([0, 1, 0, 0, 0, 0])
    elif col_diff > row_diff and row_diff > 0:
        action = np.array([0, 0, 1, 0, 0, 0])
    elif col_diff < row_diff and col_diff < 0:
        action = np.array([0, 0, 0, 1, 0, 0])

    return action

# def check_if_invalid(self, action: np.array, field: np.array) -> np.array:

#     return new_action