import ezdxf
from ezdxf.math import bulge_center, bulge_to_arc
from shapely.geometry import MultiPoint, Point, Polygon
import numpy as np

import math

CONVERT_CIRCLE_TO_POINT_DIAMETER = 0.1

def print_polyline(e):

    print("POLYLINE on layer: %s\n" % e.dxf.layer)

    # with e.points() as points:
    #     for point in points:
    #         print(point)

    print("flags    : {}".format(e.dxf.flags))
    print("closed   : {}".format(e.closed))
    print("count    : {}".format(e.dxf.count))
    print("has arc  : {}".format(e.has_arc))

    # print("start point: {},{}".format(e.x, e.y))
    # print("end point: %s\n" % e.dxf.y)

def print_line(e):

    print("LINE on layer: %s\n" % e.dxf.layer)

def print_circle(e):

    print("CIRCLE on layer: %s\n" % e.dxf.layer)
    print("center   : {}".format(e.dxf.center))
    print("radius   : {}".format(e.dxf.radius))

    # print("start point: {},{}".format(e.x, e.y))
    # print("end point: %s\n" % e.dxf.y)

def print_point(e):

    print("POINT on layer: %s\n" % e.dxf.layer)
    print("location : {}".format(e.dxf.location))

def print_arc(e):

    print("ARC on layer: %s\n" % e.dxf.layer)
    print("location : {}".format(e.dxf.location))

def dxf_lines_to_points(lines):

    output_points = []

    for i in range(0, len(lines)):

        line = lines[i]

        if i == len(lines)-1:
            next_line = lines[0]
        else:
            next_line = lines[i+1]

        if line[4] == 0.0: # straight line
            output_points.append([line[0], line[1]])
        else: # line with bulge
            start = [line[0], line[1]]
            end = [next_line[0], next_line[1]]

            output_points += dxf_bulge_to_points(start, end, line[4])

    return output_points


# def _points_to_lines(points, close=False):

#     lines = []

#     for i in range(0, len(points)-1):
#         lines.append([points[i], points[i+1]])

#     if close:
#         lines.append([points[-1], points[0]])

#     return lines


def dxf_bulge_to_points(start_point, end_point, bulge_value, degree=0.5):
    center, start_angle, end_angle, radius = bulge_to_arc(start_point, end_point, bulge_value)

    points = []

    # print("---")
    # print("bulge value: {}".format(bulge_value))
    # print("start: {} | end: {}".format(start_point, end_point))
    # print("start: {} | end: {}".format(start_angle, end_angle))
    # print("start: {} | end: {}".format(np.rad2deg(start_angle), np.rad2deg(end_angle)))

    start   = np.rad2deg(start_angle)
    end     = np.rad2deg(end_angle)

    # print("start: {} | end: {}".format(start, end))

    # bulge value positive = counter-clockwise
    # bulge value negative = clockwise

    orientation = 1
    if bulge_value < 0:
        orientation = -1

    # bulge_to_arc() switches start and end        

    if bulge_value < 0:
        tmp     = start
        start   = end
        end     = tmp

    degrees_arc = (start - end) % 360 # length of arc in degrees

    if bulge_value > 0:
        degrees_arc = (360 - degrees_arc) % 360

    num_steps = degrees_arc / degree

    for i in range(0, int(num_steps)):
        angle = (start % 360 + (degree * i * orientation)) % 360
        rad = np.deg2rad(angle)
        points.append([center[0] + radius*math.cos(rad), center[1] + radius*math.sin(rad)])
        # print("{} | {}".format(angle, points[-1]))

    return points


def dxf_circle_to_points(e, num_segments=256):

    center = e.dxf.center
    radius = e.dxf.radius

    points = []

    for i in range(0, num_segments):
        points.append(Point([
            center[0] + radius * math.cos(i*(2*math.pi/num_segments)), 
            center[1] + radius * math.sin(i*(2*math.pi/num_segments))
        ]))

    return points

def read(filename):

    doc = ezdxf.readfile(filename)

    # iterate over all entities in modelspace
    msp = doc.modelspace()

    # entity query for all LINE entities in modelspace
    # for e in msp.query('LWPOLYLINE'):
    #     print_polyline(e)
    # for e in msp.query('LINE'):
    #     print_line(e)
    # for e in msp.query('CIRCLE'):
    #     print_circle(e)
    # for e in msp.query('POINT'):
    #     print_point(e)
    # for e in msp.query('ARC'):
    #     print_arc(e)

    # ---

    polys = []
    initpoints = []

    # Lines
    for e in msp.query("LWPOLYLINE"):
        points = dxf_lines_to_points(e)
        polys.append(Polygon(points))

    # for e in msp.query("LINE"):
    #     points = dxf_lines_to_points(e)
    #     polys.append(Polygon(points))

    for e in msp.query("CIRCLE"):    

        if e.dxf.radius <= CONVERT_CIRCLE_TO_POINT_DIAMETER / 2 + 0.01:
            # initpoints.append(Point(e.dxf.center[0:2]))
            initpoints.append(e.dxf.center[0:2])
        else:
            points = dxf_circle_to_points(e)
            polys.append(Polygon(points))

    return (polys, initpoints)


TEST_FILE = "testcases/circle.dxf"
TEST_FILE = "testcases/rect.dxf"
TEST_FILE = "example_shapes/bone.dxf"
TEST_FILE = "example_shapes/rect.dxf"
TEST_FILE = "example_shapes/circle_with_cutout.dxf"
TEST_FILE = "testcases/circle_rect_cutout.dxf"
TEST_FILE = "testcases/circle_rect_cutout_points.dxf"
TEST_FILE = "SeedmarkerSpeaker_no_cutout.dxf"

if __name__ == "__main__":
    print(read(TEST_FILE))