import time
import cv2
import cv2.aruco as A
import numpy as np

import json

# unit: m
SQUARE_SIZE = .040
MARKER_SIZE = .020

NUM_HORIZONTAL = 4
NUM_VERTICAL = 6

FILENAME = "charuco.png"

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(NUM_HORIZONTAL, NUM_VERTICAL, SQUARE_SIZE, MARKER_SIZE, dictionary)

img = board.draw((
    int((NUM_HORIZONTAL*SQUARE_SIZE*1000)/2.54*75), 
    int((NUM_VERTICAL*SQUARE_SIZE*1000)/2.54*75)
))

cv2.imwrite(FILENAME, img)
print("written {} to disk. physical dimensions: {} x {} mm".format(FILENAME, NUM_HORIZONTAL*SQUARE_SIZE*1000, NUM_VERTICAL*SQUARE_SIZE*1000))

VIDEO_FILE = "IMG_3501.MOV"
SKIP = 50

OUTPUT_FILE = "calibration_iphonexs.json"

# Start capturing images for calibration
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(VIDEO_FILE)

allCorners = []
allIds = []
decimator = 0

# for i in range(300):
while True:

    ret, frame = cap.read()

    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0])>0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%SKIP==0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    decimator+=1

cap.release()
cv2.destroyAllWindows()
imsize = gray.shape

#Calibration fails for lots of reasons. Release the video if we do
try:
    cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal

    calibration_data = {}
    calibration_data["cameraMatrix"] = cameraMatrix.tolist()
    calibration_data["distCoeffs"] = distCoeffs.tolist()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(calibration_data, f)
        print("saved calibration data to {}".format(OUTPUT_FILE))

    cap = cv2.VideoCapture(VIDEO_FILE)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, dictionary)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cameraMatrix, distCoeffs)
        
        if ids is not None:
            for i in range(0, len(ids)):
                cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], MARKER_SIZE)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("failed: {}".format(e))
finally:
    cap.release()
    cv2.destroyAllWindows()