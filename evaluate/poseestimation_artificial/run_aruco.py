import time
import cv2
import cv2.aruco as A
import numpy as np
import os

import json

# unit: m
MARKER_SIZE         = 0.070
CALIBRATION_FILE    = "../calibration_blender.json"
INPUT_DIR           = "../artificial_images/aruco" 
OUTPUT_FILE         = "aruco_pose.json"
EXTENSION           = "png"

camera_matrix = None
dist_coeffs = None
with open(CALIBRATION_FILE, "r") as f:
    data = json.load(f)
    camera_matrix = np.matrix(data["cameraMatrix"])

    if data["distCoeffs"] is not None:
        dist_coeffs = np.matrix(data["distCoeffs"])
    else:
        dist_coeffs = None

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

rvecs = []
tvecs = []

input_images = []

for _, _, filenames in os.walk(INPUT_DIR):
    for fname in filenames:
        if fname.endswith(EXTENSION):
            input_images.append((INPUT_DIR, fname))

input_images = sorted(input_images)

for filename in input_images:

    frame = cv2.imread(os.path.join(*filename), cv2.IMREAD_ANYCOLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dictionary)

    # print(markerIds)

    if len(markerCorners) > 0:

        # cv2.aruco.drawDetectedMarkers(gray, markerCorners, markerIds)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, MARKER_SIZE, camera_matrix, dist_coeffs)
        rvecs.append(rvec[0].tolist())
        tvecs.append(tvec[0].tolist())

    else:
        rvecs.append(None)
        tvecs.append(None)

    # cv2.imshow('frame',gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cv2.destroyAllWindows()

with open(OUTPUT_FILE, "w") as f:

    data = {
        "rvec": rvecs,
        "tvec": tvecs
    }

    json.dump(data, f)
