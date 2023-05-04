import cv2
from ultralytics import YOLO
from multiprocessing import Process, Queue
from math import tan
import numpy as np

def object_detection_and_display(queue, bbox_queue, distance_queue):
    pth = "G:\\programming\\Kivy-App\\runs\\detect\\train5\\weights\\best.pt"
    model = YOLO(pth)
    model.fuse()
    model.overrides["verbose"] = False

    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.resizeWindow("preview", 1280, 720)
    
    fov = 1.5708
    width = 1920
    height = 1080
    f = (width / 2) / tan(fov / 2)
    K = [
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ]
    K_inv = np.linalg.inv(K)
    
    translation_vector = np.array([-30.0, 0.0, 15.0])
    axis = np.array([0, -1, 0])
    angle = -0.6981
    rotation_vector = axis * angle
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    extrinsics_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    homogeneous_matrix = np.vstack((extrinsics_matrix, np.array([0, 0, 0, 1])))
    extrinsics_matrix_inv = np.linalg.inv(homogeneous_matrix)
    
    while True:
        image = queue.get()

        if image is None:
            break

        results = model.predict(image, imgsz=(640, 640), conf=0.4)
        res_plotted = results[0].plot()
        
        for result in results:
            # print(result.cpu().numpy().boxes)
            for box in result.boxes:
                # print(box)
                # print(box.xywh)
                bbox = box.cpu().numpy().xywh[0]
                # print(f"conf:{box.cpu().numpy().conf} cls:{box.cpu().numpy().cls}")
                # print(f"xywh:{bbox}")
                if bbox_queue.full():
                    bbox_queue.get()
                bbox_queue.put(bbox)
                
                # find center
                u, v = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                bbox_center = np.array([u, v, 1])
                                
                h_obj = 0.5  # Object height (0.5 m)
                h_img = bbox[3]  # Height of the object in the image
                Z = f * h_obj / h_img
                if not distance_queue.empty():
                    Z = distance_queue.get()
                
                camera_coord = Z * K_inv @ bbox_center
                # print("camera_coord", camera_coord)
                
                world_coord = extrinsics_matrix_inv @ np.hstack((camera_coord, 1))
                # print("world_coord", world_coord)
                
                world_coord = [round(pos, 2) for pos in world_coord]

                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(res_plotted, f"{world_coord}, {Z}", (10, 30), font, 1, (255, 255, 255), 2)
                # cv2.putText(res_plotted, f"{new}", (10, 70), font, 1, (255, 255, 255), 2)
                # cv2.putText(res_plotted, f"{u} {v}", (10, 110), font, 1, (255, 255, 255), 2)
            
            # print("-----")

        resized = cv2.resize(res_plotted, (1280, 720))
        cv2.imshow("preview", resized)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
