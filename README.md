# Project Submission for the Bomberman AI Agent

This project implements a Deep Q-Network (DQN) agent for playing the Bomberman game. The agent uses reinforcement learning to improve its performance over time.

## Project Structure

The project is organized as follows:

- `Dockerfile`: Contains instructions for setting up the environment.
- `main.py`: The main script to run the game (provided by the Bomberman framework).
- `agent_code/<agent_name>/`: Contains the implementation of the DQN agent.
  - `callbacks.py`: Defines the agent's behavior and neural network structure.
  - `train.py`: Implements the training loop and reinforcement learning algorithms.

## How to recreate the experiments written in the report

- You can also find the raw data and plots in the `agent_code/dqn_agent/` folder.
- Change the parameters as mentioned in the report and run the following commands in the project root directory:

### Prerequisites

- Docker

### Building the Docker Image

To build the Docker image, run the following command in the project root directory:

```bash
docker build -t bomberman-dqn .
```

### Running the Game

#### Available Agents

- `q_table_agent`: The Q-table agent implemented in by us.
- `dqn_agent`: The DQN agent implemented in by us.
- `rule_based_agent`
- `fail_agent`
- `random_agent`
- `user_agent`
- `peaceful_agent`

#### Scenarios

- `coin-heaven`: A scenario with coins and a single exit.
- `classic`: A scenario with a bomberman and a single exit.
    - You can also remove the `--scenario <scenario-name>` flag to run the agent on classic scenario.

To run the game with the DQN agent, use the following command:

```bash
docker run -it bomberman-dqn python main.py play --my-agent <agent_name> --scenario <scenario-name>
```

### Training the Agent

To train the agent, run the following command:

```bash
docker run -it bomberman-dqn python train.py
```

This command:

```bash
docker run -it bomberman-dqn python train.py --train 1 --no-gui --scenario <scenario-name> --n-rounds 10000 --seed 100
```


- Runs the agent in training mode
- Uses the "<scenario-name>" scenario
- Trains for 10,000 rounds
- Sets a random seed of 100 for reproducibility

You can modify these parameters as needed.

## Agent Implementation

The DQN (final version) agent is implemented in the `agent_code/<agent_name>/` directory:

- `callbacks.py` contains the `DQN` class (neural network structure), `setup` function (initializes the agent), `act` function (chooses actions), and `state_to_features` function (preprocesses the game state).
- `train.py` implements the training loop, including experience replay, target network updates, and epsilon-greedy exploration.

The agent uses a replay buffer to store experiences and learns from them in batches. It also employs a target network to stabilize training.

## Monitoring Training

The training progress is logged to a CSV file named `training_data.csv`. This file contains information about average loss, rewards, Q-values, and other metrics for each round of training.

## Saving and Loading the Model

The trained model is automatically saved as `my-saved-model.pt` at the end of each round. You can use this saved model for evaluation or to continue training from a checkpoint.

## Customization

You can modify the reward structure, network architecture, or hyperparameters by editing the respective files in the `agent_code/<agent_name>/` directory.
