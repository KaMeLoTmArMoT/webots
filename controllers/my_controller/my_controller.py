from controller import Robot, DistanceSensor, Motor, GPS
import sys
import time
from random import randint, choice

print(sys.executable)

TIME_STEP = 128

robot = Robot()

gps = robot.getDevice("MY_GPS")
gps.enable(TIME_STEP)

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

speed = 5.0

while robot.step(TIME_STEP) != -1:
    gps_position = gps.getValues()
    gps_position = [round(pos, 2) for pos in gps_position]
    print(f"GPS position: {gps_position}")

    # init speeds
    left_speed = speed
    right_speed = speed

    if avoid_obstacle_counter > 0:
        avoid_obstacle_counter -= 1
        
        left_speed = speed
        right_speed = -speed
        # print(avoid_obstacle_counter)
    else:
        # read sensors outputs
        ds_values = [sensor.getValue() for sensor in ds]

        # increase counter in case of obstacle
        if ds_values[0] < 950.0 or ds_values[1] < 950.0:
            avoid_obstacle_counter = int(randint(80, 120) / speed)

        # print(ds_values)

    # write actuators inputs
    wheels[0].setVelocity(left_speed)
    wheels[2].setVelocity(left_speed)

    wheels[1].setVelocity(right_speed)
    wheels[3].setVelocity(right_speed)