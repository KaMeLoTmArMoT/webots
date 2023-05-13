import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# Define the network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 24)  # 5 inputs (speed + 4 distances), 24 outputs
        self.fc2 = nn.Linear(24, 48)  # 24 inputs, 48 outputs
        self.fc3 = nn.Linear(48, 2)  # 48 inputs, 2 outputs (speed change, angle)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the agent
class Agent:
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        print()

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def act(self, state):
        print(f"{self.epsilon=}")
        if np.random.rand() <= self.epsilon:
            print(1111111111111)
            return np.array([0.0, 0.0])
        act_values = self.model(state)
        print(22222222222222)
        return act_values.detach().numpy()  # Return the model's actions

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()

            target_f = self.model(state)
            target_f[0] = target  # Update the target for speed change
            target_f[1] = target  # Update the target for angle change

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        print("SAVING")
        with open(filename, "wb") as f:
            torch.save(self.model.state_dict(), f)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()


def compute_reward(state, action, next_state, movement):
    # speed, distances = state[0], state[1:]
    speed = 2
    distances = state
    print("compute_reward")
    print(f"{state=}, {next_state=}")
    # next_speed, next_distances = next_state[0], next_state[1:]
    next_speed = 2
    next_distances = next_state

    # Avoid collisions
    min_distance = min(next_distances)
    collision_threshold = 2  # This depends on your specific task
    if min_distance < collision_threshold:
        collision_penalty = -150 / (min_distance / collision_threshold)
        reward = collision_penalty
    else:
        collision_penalty = 150
        reward = collision_penalty

    print(f"{collision_penalty=}")

    # Maintain desired speed
    desired_speed = 2  # This depends on your specific task
    speed_deviation = abs(next_speed - desired_speed)
    reward -= speed_deviation

    if movement > 0.035:
        reward += 100 * (movement * 100)
    elif movement < 0.01:
        reward -= 2 / movement if movement != 0 else 1500

    # left and right are 1 and 3 ids
    if state[1] > 20 or state[3] > 20:
        side_distance_penalty = 100
        print(f"{side_distance_penalty=}")
        reward -= side_distance_penalty

    # Smooth navigation
    speed_change = abs(next_speed - speed)
    reward -= speed_change

    # prevent rapid direction change
    speed_change, angle_change = action
    if abs(angle_change) > 60:
        rotation_penalty = abs(angle_change) * 0.45
        print(f"{rotation_penalty=}")
        reward -= rotation_penalty
    else:
        reward += 200

    for dist in state:
        if dist == 3000:
            reward -= 150

    # Forward progress - assuming that the first sensor is the forward-facing one
    # forward_distance_change = max(distances[0] - next_distances[0], 0)
    # print(f"{forward_distance_change=}")
    # reward += forward_distance_change

    return reward


def main_loop():
    # Main loop
    agent = Agent()
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = np.random.rand(1, 5)  # Initialize state, in practice, you'd get this from the environment
        state = torch.from_numpy(state).float()
        for time in range(500):
            action = agent.act(state)
            next_state = np.random.rand(1, 5)  # In practice, this would be the new state after taking the action
            next_state = torch.from_numpy(next_state).float()
            # reward =  -np.sum(next_state.numpy())  # This is a placeholder reward function
            reward = compute_reward(state, action, next_state)
            agent.remember(state, action, reward, next_state)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


if __name__ == "__main__":
    main_loop()
