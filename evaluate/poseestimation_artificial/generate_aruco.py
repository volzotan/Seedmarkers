import cv2

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
img = cv2.aruco.drawMarker(dictionary, 1, 1000)
cv2.imwrite("aruco_4x4_50_01.png", img)
