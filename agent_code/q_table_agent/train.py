from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import pandas as pd
import events as e
from .callbacks import state_to_features, MODEL_NAME

# Define the Transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 0.1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

PLACEHOLDER_EVENT = "PLACEHOLDER"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.CSV_FILE = "2-coin-heaven-1000.csv"
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.losses = []
    self.loss_df = pd.DataFrame(columns=["Round", "Step", "Loss"])
    self.round_count = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    if old_game_state is None:
        return;

    action_index = ACTIONS.index(self_action)
 
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    reward = reward_from_events(self, events)
 
    self.transitions.append(Transition(old_features, action_index, new_features, reward))

    old_q_value = self.q_table[
        int(old_features[0]), int(old_features[1]),
        int(old_features[2]), int(old_features[3]),
        int(old_features[4]), action_index
    ]

    future_rewards = np.max(self.q_table[
        int(new_features[0]), int(new_features[1]),
        int(new_features[2]), int(new_features[3]),
        int(new_features[4])
    ])

    td_error = reward + DISCOUNT_FACTOR * future_rewards - old_q_value
    loss = td_error ** 2
    self.losses.append(loss)
    self.loss_df["Round"] = new_game_state["round"]
    self.loss_df["Step"] = new_game_state['step']
    self.loss_df["Loss"] = loss

    new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_rewards)
 
    self.q_table[
        int(old_features[0]), int(old_features[1]),
        int(old_features[2]), int(old_features[3]),
        int(old_features[4]), action_index
    ] = new_q_value


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_features = state_to_features(self, last_game_state)
 
    reward = reward_from_events(self, events)

    action_index = ACTIONS.index(last_action)
    
    self.transitions.append(Transition(last_features, action_index, None, reward))

    old_q_value = self.q_table[
        int(last_features[0]), int(last_features[1]),
        int(last_features[2]), int(last_features[3]),
        int(last_features[4]), action_index
    ]

    new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * reward

    self.q_table[
        int(last_features[0]), int(last_features[1]),
        int(last_features[2]), int(last_features[3]),
        int(last_features[4]), action_index
    ] = new_q_value

    # Store the model
    with open(MODEL_NAME, "wb") as file:
        pickle.dump(self.q_table, file)

    avg_loss = np.mean(self.losses) if self.losses else 0
    self.logger.info(f"Average loss for this round: {avg_loss}")
    self.losses.clear()
    self.round_count += 1
    self.loss_df.to_csv(self.CSV_FILE, index=False)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.GOT_KILLED: -1.0,
        e.SURVIVED_ROUND: 2.0,
        e.INVALID_ACTION: -0.3,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
