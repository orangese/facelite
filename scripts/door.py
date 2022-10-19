import sys

sys.path.insert(1, "../")

import cv2
from facenet import FaceNet
from util.data import Database
from util.visuals import Camera


if __name__ == "__main__":
    database = Database()
    database.load("../config/database/db.json")

    facenet = FaceNet(
        model_path="../config/models/20180402-114759.tflite",
        data=database,
    )
    # data = facenet.embed_dir(
    #     img_dir="../config/imgs"
    # )
    # database.set_data(data)
    # database.dump("../config/database/db.json")

    # cap = cv2.VideoCapture("/Users/ryan/Desktop/ryandoor.mov")
    cap = Camera()
    facenet.run_on_stream(cap)
