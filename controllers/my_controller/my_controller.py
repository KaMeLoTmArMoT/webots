from math import sqrt

import numpy as np
from controller import Robot, DistanceSensor, Motor, GPS
import sys
import time
from random import randint, choice
from model import *
from scipy.spatial.distance import euclidean

BASE_SPEED = 10.0
# print(sys.executable)

TIME_STEP = 128

robot = Robot()

gps = robot.getDevice("MY_GPS")
gps.enable(TIME_STEP)

lasers = dict()
names = [
    "ds_laser forward",
    "ds_laser left",
    "ds_laser left 45",
    "ds_laser right",
    "ds_laser right 45",
]
for name in names:
    lasers[name] = robot.getDevice(name)
    print(lasers[name])
    lasers[name].enable(TIME_STEP)

avoid_obstacle_counter = 0

# Initialize distance sensors
ds = []
ds_names = ["ds_left", "ds_right"]
for name in ds_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    ds.append(sensor)

# Initialize motors
wheels = []
wheels_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
for name in wheels_names:
    motor = robot.getDevice(name)
    motor.setPosition(float("inf"))
    wheels.append(motor)

agent = Agent()
episodes = 1000
batch_size = 32


# TODO skip first 10 states


def get_state(lasers):
    distances = [el.getValue() for el in lasers.values()]
    print(distances)
    state = np.array(distances)
    state = torch.from_numpy(state).float()
    return state


run_time = 0
epoch = 0
action = agent.act(np.random.rand(1, 5))

current_state = np.random.rand(5)
current_state = torch.from_numpy(current_state).float()
print(np.shape(current_state), "original", current_state)


def apply_action(speed_change, angle_change):
    # Apply speed change to the base speed
    speed = BASE_SPEED + speed_change

    # Calculate the speed difference for the wheels
    speed_diff = angle_change * BASE_SPEED

    # Calculate individual wheel speeds
    left_speed = np.clip(speed + speed_diff / 2.0, -BASE_SPEED, BASE_SPEED)
    right_speed = np.clip(speed - speed_diff / 2.0, -BASE_SPEED, BASE_SPEED)

    return left_speed, right_speed


def distance(current, prev):
    x = (current[0] - prev[0]) ** 2
    y = (current[1] - prev[1]) ** 2
    total = sqrt(x + y)
    return total


prev_pos = gps.getValues()
angle_change_save = []
mean_reward = []
agent.load("01.12_model.tr")
while robot.step(TIME_STEP) != -1:
    print(f"\n\n--------- {epoch}|{run_time} -------")
    gps_position = gps.getValues()
    gps_position = [round(pos, 3) for pos in gps_position]
    movement = distance(gps_position, prev_pos)
    print(f"GPS position: {gps_position}, prev: {prev_pos}, {movement=}")

    print(f"{[(key, el.getValue()) for key, el in lasers.items()]}")

    run_time += 1
    if run_time == 500:
        # reset
        current_state = get_state(lasers)
        run_time = 0
        epoch += 1
        # TODO end train
        agent.save("model.tr")

        with open("log.txt", "a") as f:
            f.write(f"{sum(mean_reward) / len(mean_reward)}")
        mean_reward = []

    if run_time < 500:
        old_state = current_state
        current_state = get_state(lasers)

        print("main")
        print(f"{old_state=}, {current_state=}")

        reward = compute_reward(old_state, action, current_state, movement)
        print(f"{reward=}")
        mean_reward.append(reward)
        agent.remember(old_state, action, reward, current_state)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        action = agent.act(current_state)
        print(f"{action=}, {type(action)=}, {np.shape(action)=}")

        speed_change, angle_change = action
        left_speed, right_speed = apply_action(0, angle_change)
        angle_change_save.append(angle_change)
        print(f"~~~ {np.min(angle_change_save)=}, {np.mean(angle_change_save)}, {np.max(angle_change_save)}")

    # # init speeds
    # left_speed = speed
    # right_speed = speed

    # write actuators inputs
    wheels[0].setVelocity(left_speed)
    wheels[2].setVelocity(0)
    wheels[2].setForce(0)

    wheels[1].setVelocity(right_speed)
    wheels[3].setVelocity(0)
    wheels[3].setForce(0)

    prev_pos = gps_position
