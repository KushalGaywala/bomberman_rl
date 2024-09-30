import os
import pickle
import random
import numpy as np

# Possible actions in the game
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EXPLORATION_RATE = 0.1  # Exploration rate, defined here to maintain consistency

MODEL_NAME = "2-coin-heaven-1000.pt"

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
    if self.train or not os.path.isfile(MODEL_NAME):
        self.logger.info("Setting up model from scratch.")
        self.q_table = np.zeros((17, 17, 17, 17, 100, len(ACTIONS)))  
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_NAME, "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = EXPLORATION_RATE
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(self, game_state)
 
    agent_x, agent_y, coin_x, coin_y, coin_dist = features[:5]
    q_values = self.q_table[int(agent_x), int(agent_y), int(coin_x), int(coin_y), int(coin_dist)]

    action_idx = np.argmax(q_values)
    return ACTIONS[action_idx]

def state_to_features(self, game_state: dict) -> np.array:
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

    # Extract features like the agent's position, the positions of walls, coins, and bombs
    agent = np.array(game_state['self'][3])  # Agent's position
    coins = np.array(game_state['coins'])

    nearest_coin, nearest_coin_dist = get_nearest_coin(self, agent, coins)

    features = np.concatenate([
        agent, 
        nearest_coin, 
        np.array([nearest_coin_dist]),
    ])

    self.logger.debug(f"\nFeatures:\n{features}")

    return features

def get_nearest_coin(self, agent: np.array, coins: np.array) -> np.array:
    if coins.size <= 0:
        return np.array([0, 0]), 0

    distances = coins - agent
    nearest_coin_dist = np.min(np.sum(np.abs(distances), axis=1))
    nearest_coin = coins[np.argmin(np.sum(np.abs(distances), axis=1))]
    return nearest_coin, nearest_coin_dist
