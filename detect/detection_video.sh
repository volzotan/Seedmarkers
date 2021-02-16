#!/bin/sh

python3 detection.py                                                \
--input-video example.mov                                           \
--calibration-data ../evaluate/calibration_iphonexs.json            \
--draw-output                                                       \
--verbose                                                           \
--descriptor "012343332343323432342323|136.73:24.71:8.25|102.87:19.63:6.54|115.11:32.91:7.65|152.39:41.88:7.10|98.91:54.16:2.87|84.78:81.76:4.47|98.07:78.47:4.90|167.81:75.73:6.39|182.79:60.37:4.27|179.36:24.95:4.16|16.67:82.06:3.66|16.67:16.78:3.71"