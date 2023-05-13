import os
import cv2
import time
import numpy as np
from controller import Robot, Camera
from obj_det_display import object_detection_and_display
from multiprocessing import Process, Queue


def range_sensor(bbox_queue, lidar_sensor, distance_queue):
    try:
        if not bbox_queue.empty():
            range_image = lidar_sensor.getRangeImage()
            # print(np.shape(range_image))

            range_image_copy = np.copy(range_image)
            inf_mask = np.isinf(range_image_copy)
            range_image_copy[inf_mask] = 0
            range_image_copy = range_image_copy.reshape((1080, 1920))
            range_image_copy = (range_image_copy - np.min(range_image_copy)) / (
                    np.max(range_image_copy) - np.min(range_image_copy))
            range_image_copy = (range_image_copy * 255).astype(np.uint8)

            # print(np.min(range_image_copy), np.max(range_image_copy))
            range_image_copy = cv2.resize(range_image_copy, (1280, 720))
            # cv2.imshow("Range Image", range_image_copy)

            bbox = bbox_queue.get()
            # print("Received bbox:", bbox)

            # crop the depth image to the bbox
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            depth = np.copy(range_image)
            inf_mask = np.isinf(depth)
            depth[inf_mask] = 0
            depth = depth.reshape((1080, 1920))

            depth_crop = depth[y:y + h, x:x + w]
            d = np.mean(depth_crop)
            # print(d)

            if distance_queue.full():
                distance_queue.get()  # Remove the oldest frame from the queue if it's full
            distance_queue.put(d)

            depth_crop = (depth_crop - np.min(depth_crop)) / (np.max(depth_crop) - np.min(depth_crop))
            depth_crop = (depth_crop * 255).astype(np.uint8)

            # print(np.min(depth_crop), np.max(depth_crop))
            depth_crop = cv2.resize(depth_crop, (w * 3, h * 3))
            # cv2.imshow("Depth Crop", depth_crop)
    except Exception as e:
        # print(e)
        pass


def main():
    TIME_STEP = 128

    robot = Robot()
    camera = Camera("static_camera")
    camera.enable(TIME_STEP)

    lidar_sensor = robot.getDevice("my_lidar_sensor")
    lidar_sensor.enable(TIME_STEP)

    queue = Queue(maxsize=1)
    bbox_queue = Queue(maxsize=1)
    distance_queue = Queue(maxsize=1)

    worker = Process(target=object_detection_and_display, args=(queue, bbox_queue, distance_queue,))
    worker.start()

    image_width = camera.getWidth()
    image_height = camera.getHeight()

    while robot.step(TIME_STEP) != -1:
        # check for bbox in the queue

        cv2.waitKey(1)

        image_data = camera.getImage()

        image = np.frombuffer(image_data, dtype=np.uint8).reshape((image_height, image_width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if queue.full():
            queue.get()  # Remove the oldest frame from the queue if it's full

        queue.put(image)

    # Send a None to indicate the end of the simulation
    queue.put(None)

    worker.join()


if __name__ == "__main__":
    main()
