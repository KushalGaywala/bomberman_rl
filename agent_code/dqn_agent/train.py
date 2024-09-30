from collections import namedtuple, deque
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, DQN, ACTIONS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
from time import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

LOG_INTERVAL = 100
SAVE_INTERVAL = 10000

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.replay_buffer = deque(maxlen=TRANSITION_HISTORY_SIZE)
    input_size = 17 * 17 * 6 
    self.model = DQN(input_size=input_size, output_size=len(ACTIONS))
    self.target_model = DQN(input_size=input_size, output_size=len(ACTIONS))
    self.target_model.load_state_dict(self.model.state_dict())
    self.train_model = train_model
    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.total_steps = 0
    self.epsilon = EPSILON_START

    self.log_buffer = deque(maxlen=SAVE_INTERVAL // LOG_INTERVAL)
    self.last_log_time = time()

    self.log_count = 0

    self.round_losses = []
    self.round_rewards = []
    self.round_q_values = []
    self.round_start_time = time()

    self.csv_file = open('training_data.csv', 'w', newline='')
    self.csv_writer = csv.writer(self.csv_file)
    self.csv_writer.writerow(['Round', 'Avg_Loss', 'Avg_Reward', 'Avg_Q_Value', 'Epsilon', 'Duration', 'Total_Steps'])

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

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)
    
    old_state = state_to_features(old_game_state)
    action = ACTIONS.index(self_action)
    new_state = state_to_features(new_game_state)
    
    self.replay_buffer.append(Transition(old_state, action, new_state, reward))
    
    if len(self.replay_buffer) >= BATCH_SIZE:
        self.train_model(self)



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

    reward = reward_from_events(self, events)
    
    last_state = state_to_features(last_game_state)
    action = ACTIONS.index(last_action)
    
    self.replay_buffer.append(Transition(last_state, action, None, reward))
    
    if len(self.replay_buffer) >= BATCH_SIZE:
        self.train_model(self)

    torch.save(self.model.state_dict(), "my-saved-model.pt")
    
    round_duration = time() - self.round_start_time
    avg_loss = np.mean(self.round_losses) if self.round_losses else 0
    avg_reward = np.mean(self.round_rewards) if self.round_rewards else 0
    avg_q_value = np.mean(self.round_q_values) if self.round_q_values else 0

    self.csv_writer.writerow([
        last_game_state['round'],
        avg_loss,
        avg_reward,
        avg_q_value,
        self.epsilon,
        round_duration,
        self.total_steps
    ])
    self.csv_file.flush()

    self.round_losses = []
    self.round_rewards = []
    self.round_q_values = []
    self.round_start_time = time()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -0.3,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.5,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def train_model(self):
    self.total_steps += 1

    batch = random.sample(self.replay_buffer, BATCH_SIZE)
    states, actions, next_states, rewards = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
    non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None]))

    q_values = self.model(states).gather(1, actions.unsqueeze(1))
    next_q_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_q_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
    expected_q_values = rewards + (GAMMA * next_q_values)

    loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if self.total_steps % TARGET_UPDATE == 0:
        self.target_model.load_state_dict(self.model.state_dict())

    current_time = time()
    log_entry = {
        'Step': self.total_steps,
        'Loss': loss.item(),
        'Epsilon': self.epsilon,
        'Reward': rewards.mean().item(),
        'Q_Value_Mean': q_values.mean().item(),
        'Q_Value_Max': q_values.max().item(),
        'Time': current_time - self.last_log_time
    }
    self.log_buffer.append(log_entry)
    self.last_log_time = current_time

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    self.round_losses.append(loss.item())
    self.round_rewards.append(rewards.mean().item())
    self.round_q_values.append(q_values.mean().item())