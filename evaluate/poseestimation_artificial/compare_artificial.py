import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from datetime import datetime

import math
import numpy as np

import cv2

np.set_printoptions(precision=3, suppress=True)

plt.style.use("grayscale")
# plt.style.use("despat.mplstyle")
# plt.style.use("despat_dark.mplstyle")

# COLOR_1 = "#29798e" #"#3b528b"
# COLOR_2 = "#fde725" #"#81d34d"
COLOR_1 = "#3b528b"
COLOR_2 = "#81d34d"
COLOR_3 = "#fde725"

rvecs = {}
tvecs = {}


def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])

    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

with open("../artificial_images/ground_truth.json", "r") as f:
    data = json.load(f)

    rvecs["gt"] = []
    tvecs["gt"] = []

    for key in data.keys():
        rvecs["gt"] += [data[key]["rvec"]]
        tvecs["gt"] += [data[key]["tvec"]]

# with open("apriltag_pose.json", "r") as f:
#     data = json.load(f)

#     rvecs["apriltag"] = data["rvec"]
#     tvecs["apriltag"] = data["tvec"]

with open("aruco_pose.json", "r") as f:
    data = json.load(f)

    rvecs["aruco"] = data["rvec"]
    tvecs["aruco"] = data["tvec"]

with open("seedmarker_pose_90.json", "r") as f:
    data = json.load(f)

    rvecs["seedmarker_90"] = data["rvec"]
    tvecs["seedmarker_90"] = data["tvec"]

with open("seedmarker_pose_180.json", "r") as f:
    data = json.load(f)

    rvecs["seedmarker_180"] = data["rvec"]
    tvecs["seedmarker_180"] = data["tvec"]


num_frames = len(rvecs["aruco"])

print("total frames: {}".format(num_frames))

fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(8, 4))

xs = list(range(0, num_frames))
colors = [COLOR_1, COLOR_2, COLOR_3]
# markernames = ["aruco", "apriltag", "seedmarker"]
markernames = ["aruco", "seedmarker_90"] #, "seedmarker_180"]
for i in range(0, len(markernames)):

    t_ys = []
    r_ys = []
    for j in range(0, num_frames):
        tvec = tvecs[markernames[i]][j]
        rvec = rvecs[markernames[i]][j]

        if tvec is None:
            t_ys.append(None)
            r_ys.append(None)
            continue

        tmat = np.matrix(tvec)
        if tmat.shape == (3, 1):
            tmat = tmat.transpose()

        rmat = np.matrix(rvec)
        if rmat.shape == (3, 1):
            rmat = rmat.transpose()

        if rmat.shape == (3, 3):
            r = rmat # apriltags returns rotation matrix 
            t = -r.transpose() * tmat.transpose()
            t = t.tolist()
        else:
            r, _ = cv2.Rodrigues(rmat)
            t = -r.transpose() * tmat.transpose()
            t = t.tolist()

        # r2, _ = cv2.Rodrigues(r.transpose())

        # coordinate system change, opencv expects Z to face to camera
        gtx = tvecs["gt"][j][0]
        gty = tvecs["gt"][j][2] * -1
        gtz = tvecs["gt"][j][1] * -1

        grx = rvecs["gt"][j][0] - 1*math.pi
        gry = rvecs["gt"][j][2]
        grz = rvecs["gt"][j][1] * -1

        r_deg = 180*rotationMatrixToEulerAngles(r)/math.pi
        gr = 180*np.matrix([grx, gry, grz])/math.pi

        t_ys.append(
            abs(t[0][0]-gtx) + 
            abs(t[1][0]-gty) +
            abs(t[2][0]-gtz)  
        )

        r_ys.append(
            # abs(r_deg[0]-gr[0, 0]) + 
            abs(r_deg[1]-gr[0, 1]) #+
            # abs(r_deg[2]-grz)  
        )

    axs[0].plot(xs, t_ys, linestyle="solid", marker=None, color=colors[i])
    axs[1].plot(xs, r_ys, linestyle="solid", marker=None, color=colors[i])

axs[0].set_ylabel('Translation error [m]')
axs[0].set_xlim([0, num_frames])
axs[0].set_ylim([0, 1.0])

axs[1].set_ylabel('Rotation error [deg]')
axs[1].set_xlabel('Frame')
axs[1].set_ylim([0, 10])

custom_lines = [Line2D([0], [0], color=COLOR_1, lw=4),
                Line2D([0], [0], color=COLOR_2, lw=4),
                Line2D([0], [0], color=COLOR_3, lw=4)]

axs[0].legend(custom_lines, markernames, frameon=False)

fig.tight_layout()
plt.savefig('plot_compare.png', transparent=False)
