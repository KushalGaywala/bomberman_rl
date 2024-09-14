from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features

# Define the Transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 100  # Number of transitions to keep for experience replay
LEARNING_RATE = 0.1            # Alpha: The learning rate for Q-learning
DISCOUNT_FACTOR = 0.99         # Gamma: The discount factor for future rewards
EXPLORATION_RATE = 0.1         # Epsilon: The probability of taking a random action (exploration)

# Possible actions in the game
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def setup_training(self):
    """
    Initialize self for training purposes.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Update the agent's model based on the events that occurred.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    if old_game_state is None:
        return;

    # Convert action to index
    action_index = ACTIONS.index(self_action)
    
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    # Calculate reward from the occurred events
    reward = reward_from_events(self, events, old_features)
    
    # Append transition to the deque
    self.transitions.append(Transition(old_features, action_index, new_features, reward))

    # Q-learning update
    old_q_value = self.q_table[
        int(old_features[0]), int(old_features[1]),  # Agent's position
        int(old_features[2]), int(old_features[3]),  # Nearest coin's position
        int(old_features[4]),  # Distance to the nearest coin
        action_index
    ]
    future_rewards = np.max(self.q_table[
        int(new_features[0]), int(new_features[1]),  # Agent's position
        int(new_features[2]), int(new_features[3]),  # Nearest coin's position
        int(new_features[4])   # Distance to the nearest coin
    ])  # Max Q-value for next state

    # Q-learning formula: Q(s, a) = (1 - alpha) * Q(s, a) + alpha * [reward + gamma * max(Q(s', a'))]
    new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_rewards)
    
    # Update Q-table
    self.q_table[
        int(old_features[0]), int(old_features[1]),
        int(old_features[2]), int(old_features[3]),
        int(old_features[4]), action_index
    ] = new_q_value


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Final update at the end of the round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the final transition
    last_features = state_to_features(self, last_game_state)
    
    # Calculate reward for the last action
    reward = reward_from_events(self, events, last_features)

    # Convert action to index
    action_index = ACTIONS.index(last_action)
    
    self.transitions.append(Transition(last_features, action_index, None, reward))

    # Q-learning update for the last action
    old_q_value = self.q_table[
        int(last_features[0]), int(last_features[1]),
        int(last_features[2]), int(last_features[3]),
        int(last_features[4]), action_index
    ]
    new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * reward  # No future rewards since the game ended
    
    # Update Q-table
    self.q_table[
        int(last_features[0]), int(last_features[1]),
        int(last_features[2]), int(last_features[3]),
        int(last_features[4]), action_index
    ] = new_q_value

    # Save the updated model
    with open("winston_zeddmore.pt", "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(self, events: List[str], last_features) -> int:
    """
    Assign rewards based on the events that occurred.
    """
    game_rewards = {
        e.BOMB_DROPPED: 1,
        e.BOMB_EXPLODED: 1,
        e.COIN_COLLECTED: 10,
        e.COIN_FOUND: 1000,
        e.CRATE_DESTROYED: 10,
        e.INVALID_ACTION: -1000,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -10,
        e.MOVED_UP: -100,
        e.MOVED_RIGHT: -100,
        e.MOVED_DOWN: -100,
        e.MOVED_LEFT: -100,
        e.WAITED: -10,
        "OPTIMAL_ACTION": 1000,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

        if (last_features[5] == 1 and event == e.MOVED_UP) or \
           (last_features[6] == 1 and event == e.MOVED_RIGHT) or \
           (last_features[7] == 1 and event == e.MOVED_DOWN) or \
           (last_features[8] == 1 and event == e.MOVED_LEFT) or \
           (last_features[9] == 1 and event == e.WAITED) or \
           (last_features[10] == 1 and event == e.BOMB_DROPPED):
            reward_sum += game_rewards["OPTIMAL_ACTION"]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
