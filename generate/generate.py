import powerwrapper
from writer import *

import logging as log
import math
import os
from datetime import datetime
import random
import argparse
from operator import attrgetter

import dxfread
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint, GeometryCollection
from shapely import ops, affinity

MIN_ITERATIONS              = 10
MAX_ITERATIONS              = 10000
MIN_ITERATIONS_BEFORE_SCALING = 1

GRAVITY_PULL_RATE           = 0.1
CENTROID_CHANGE_RATE        = 0.03 # 0.1
SCALING_FACTOR_CHANGE_RATE  = 1.5 # 2.0 # 3.0
MIN_ERROR                   = 1e-7 # 1e-10

CHECK_FOR_COLLISION         = True

SIMPLIFICATION_MAX_ERROR    = 0.1
SIMPLIFICATION_ERROR_CUTOUT = 0.5

LARGEST_CIRCLE_PRECISION    = 0.4 # units = mm

MIN_FEATURE_SIZE            = 1.2
LINE_WIDTH                  = 2.4
OVERSHRINK                  = 2.0 # 2.5
ROOT_TREE_SHRINK            = LINE_WIDTH

WEIGHTING_DEPTH_FACTOR      = 0.2

LEAFS_AS_CIRLCES            = True
NO_DUMMY                    = True

EXPORT_DEBUG_IMAGE_SKIP     = 1000
EXPORT_DEBUG_IMAGE          = True
EXPORT_DEBUG_CLEAR_IMAGE    = True
DEBUG_DRAW_VORONOI_CELLS    = True

OUTPUT_NAME                 = "{:05}.png"
OUTPUT_IMAGE_DIMENSIONS     = [1000, 1000]
FIXED_OUTPUT_SCALE_FACTOR   = None
OUTPUT_PADDING              = 50

font                        = ImageFont.load_default()
font_large                  = ImageFont.truetype("FiraMono-Regular.ttf", 16)
font_large_bold             = ImageFont.truetype("FiraMono-Bold.ttf", 16)

ERROR_ILLEGAL_INPUT             = 0x100 << 1
ERROR_ILLEGAL_PARAM             = 0x100 << 2
ERROR_EMPTY_BOUNDARY_SUBTREE    = 0x100 << 3
ERROR_EMPTY_TREE                = 0x100 << 4
ERROR_INVALID_GEOMETRY          = 0x100 << 5
ERROR_FAILED_VALIDATION         = 0x100 << 6

class Subtree(object):

    def __init__(self, tree, boundary, config, gravity_vector=None, cutouts=[], init_points=[], depth=0):
        
        self.tree = tree
        self.subtree_names = self.generate_subtree_names(self.tree)
        self.depth = depth

        self.cutout_list = cutouts
        self.cutouts = []
        for c in self.cutout_list:

            # if cutout depth index matches this sublayer
            # will create issues when this depth number has multiple subtrees (e.g. 2 and 0122)

            if c[0] == self.depth:
                self.cutouts.append(c[1])

        log.info("---")
        log.info("node level: {} | name: {}".format(self.tree[0], "".join(self.tree)))
        log.info("subtrees: {}".format(["".join(x) for x in self.subtree_names]))

        self.boundary = boundary
        self.config = config

        if type(self.boundary) is MultiPolygon:
            log.error("boundary is MultiPolygon!")
            exit(ERROR_INVALID_GEOMETRY)
        if not self.boundary.is_valid:
            log.error("boundary is not valid")
            exit(ERROR_INVALID_GEOMETRY)
        if self.boundary.is_empty:
            log.error("boundary polygon is empty: {}".format(self.boundary))
            exit(ERROR_INVALID_GEOMETRY)
        if self.boundary.area <= 0.5:
            log.error("boundary too small: {}".format(self.boundary))
            exit(ERROR_INVALID_GEOMETRY)

        if self.depth == 0:
            shrinkage = ROOT_TREE_SHRINK
        else:
            shrinkage = self.config["boundary_shrink"]
        self.boundary_shrinked = self.boundary.buffer(-OVERSHRINK-shrinkage).buffer(OVERSHRINK)

        if type(self.boundary_shrinked) is MultiPolygon:
            log.warning("shrinked boundary is MultiPolygon. Attempting merge...")
            self.boundary_shrinked = ops.unary_union(self.boundary_shrinked.geoms)
            self.boundary_shrinked = self.get_largest_poly_of_multipolygon(self.boundary_shrinked)

        if self.boundary_shrinked.area < 0.5:
            log.warning("shrinked boundary too small (area: {:5.2f}). Try no overshrink".format(self.boundary_shrinked.area))
            self.boundary_shrinked = self.boundary.buffer(-shrinkage)
            log.warning("shrinked boundary area without overshrink: {:5.2f}".format(self.boundary_shrinked.area))

        if self.boundary_shrinked.area < 0.5:
            log.error("boundary too small: {}".format(self.boundary_shrinked))
            exit(ERROR_INVALID_GEOMETRY)

        # put holes in the boundary polygon where cutouts are located
        self.boundary_with_cutouts = self.boundary_shrinked
        for c in self.cutouts:
            self.boundary_with_cutouts = self.boundary_with_cutouts.difference(c)
        if type(self.boundary_with_cutouts) is MultiPolygon:
            self.boundary_with_cutouts = self.get_largest_poly_of_multipolygon(self.boundary_with_cutouts)

        self.centers = None
        self.old_error = None
        self.new_error = None

        # create dummy voronoi centers for the cutouts
        dummy_centers = []
        if not NO_DUMMY:
            for c in self.cutouts:
                c = c.buffer(self.config["boundary_shrink"]*2 + SIMPLIFICATION_ERROR_CUTOUT*2)
                simplified_cutout = c.simplify(SIMPLIFICATION_ERROR_CUTOUT)
                dummy_centers += self.polygon_to_points(simplified_cutout, 1.0)

            # dummy_centers += list(simplified_cutout.exterior.coords)[:-1] # coords are a closed CCW loop, remove duplicate starting point
        self.dummy_centers = np.array(dummy_centers)

        # fix cutout subtrees
        self.fixed_subtrees = []
        # remove fixed subtrees (= cutout subtrees) from the regular ones. We reintroduce them during finalization
        for i in range(0, len(self.cutouts)):
            self.fixed_subtrees.append(["*"])
        # for i in range(0, len(self.cutouts)):
        #     self.fixed_subtrees.append(self.subtree_names[i])
        # self.subtree_names = self.subtree_names[len(self.cutouts):]
        
        self.local_init_points = []
        if len(self.subtree_names) > 0:
            self.weights = np.zeros([len(self.subtree_names)], dtype=np.float)   
            for i in range(0, len(self.subtree_names)):
                self.weights[i] = self.calculate_weight_for_subtree(self.subtree_names[i])

            # Qhull fails to compute a convex hull if (lifted) points are colinear (either in XY or weight used for lifting)
            # so just add a bit of wiggle room with the weights rather than the coordinates
            for i in range(0, self.weights.shape[0]):
                self.weights[i] += random.uniform(0.005, 0.001)

            self.local_scaling_factor = np.ones_like(self.weights, dtype=np.float)

            for init_point in init_points:
                if self.boundary_with_cutouts.contains(Point(init_point)):
                    self.local_init_points.append(init_point)

            # no artificial init points for the root node
            # if self.depth == 0:
            #     self.local_init_points = []

            log.debug("init points: {} artificial, {} generated".format(
                len(self.local_init_points), 
                max(0, len(self.subtree_names) - len(self.local_init_points)))
            )
            generated_init_points = self.generate_init_points(len(self.subtree_names) - len(self.local_init_points))

            if generated_init_points is None:
                log.error("could not generate init points")
                exit(ERROR_INVALID_GEOMETRY)

            all_init_points = self.local_init_points + generated_init_points
            if len(self.subtree_names) < len(all_init_points):
                all_init_points = all_init_points[:len(self.subtree_names)]

            self.centers = np.asarray(all_init_points, dtype=np.float)

            self.old_error = np.zeros_like(self.weights, dtype=np.float)
            self.new_error = np.zeros_like(self.weights, dtype=np.float)

        else:
            # self.weights = np.ones([len(self.tree)], dtype=np.float)
            pass

        # gravity for ReacTIVision orientation vector
        self.gravity_vector = gravity_vector
        self.gravity = np.zeros([len(self.subtree_names)], dtype=np.float)
        for i in range(0, len(self.subtree_names)):
            subtree_name = self.subtree_names[i]
            
            # if len(subtree_name) <= 1: # ignore the leaves
            #     continue

            if int(max(subtree_name)) % 2 == 0:
                self.gravity[i] = 1

        self.iteration = 0
        self.polys = []

        # will be created during finalization
        self.leaf_poly = None
        self.subtrees = []
        self.finished = False


    def __lt__(self, other):
        if not isinstance(other, Subtree):
            return NotImplemented

        # comparison criterion for LHDS is the numerical greater/lesser than of the 
        # name of the subtree. Thus width beats depth, ie. 3444 comes before 345

        return int(self.get_lhds_name()) <= int(other.get_lhds_name())


    # def get_weight(self):
    #     if len(self.subtrees) == 0:
    #         return 1
    #     else:
    #         weight = 1
    #         for s in self.subtrees:
    #             weight += s.get_weight()
    #         return weight


    def get_max_depth(self, depth=0):
        depths = [depth]
        for s in self.subtrees:
            depths.append(s.get_max_depth(depth=depth+1))

        return max(depths)


    def get_lhds_name(self, depth=0):
        name = str(depth)
        subtrees_sorted = sorted(self.subtrees, reverse=True)
        for s in subtrees_sorted:
            name += s.get_lhds_name(depth+1)

        return name


    def polygon_to_points(self, poly, max_distance):

        points = []
        poly_points = list(poly.exterior.coords)

        for i in range(0, len(poly_points)-1):
            c = poly_points[i]
            n = poly_points[i+1]
            dist = np.linalg.norm(np.matrix(c)-np.matrix(n))

            points.append(c)

            if dist > max_distance:
                segments = int(dist/max_distance)
                for j in range(1, segments):
                    points.append([
                        c[0] + ((n[0]-c[0])/segments)*j,
                        c[1] + ((n[1]-c[1])/segments)*j,
                    ])

        return points


    def generate_init_points(self, num):

        if num <= 0:

            # case <= 0 point
            # only using artificial init points
            # no need to generate points

            return []

        _, center, dist = self.get_largest_circle_in_poly(self.boundary_with_cutouts)

        if center is None:
            return None

        dist *= 0.5

        points = []

        log.debug("generating init points")
        log.debug("centroid: {}".format(center))
        log.debug("boundary area: {}".format(self.boundary.area))
        log.debug("dist for init points: {}".format(dist))

        if num == 1:

            # case == 1 point

            points.append(center)

        elif num == 2:
            
            # case == 2
            # prevent points from being placed dead center on an axis 
            # (otherwise they can't wiggle their way (fast) into a position with better centricity)

            minx, miny, maxx, maxy = self.boundary.bounds
            epsilon = 0.1
            
            if abs((maxx-minx) - (maxy-miny)) > epsilon: # boundary is wide in X dim

                candidates = [
                    Point([center[0]+dist, center[1]+epsilon]),
                    Point([center[0]-dist, center[1]-epsilon])
                ]

            elif abs((maxy-miny) - (maxx-minx)) > epsilon: # boundary is high in Y dim

                candidates = [
                    Point([center[0]+epsilon, center[1]+dist]),
                    Point([center[0]-epsilon, center[1]-dist])
                ]

            else: # boundary is square-ish, 45 degree angle

                orientation = -1 ** self.depth

                candidates = [
                    Point([center[0]+(dist)*orientation,            center[1]+(dist+epsilon)*orientation]),
                    Point([center[0]-(dist-epsilon)*orientation,    center[1]-(dist)*orientation])
                ]

            for point in candidates:
                if self.boundary_with_cutouts.contains(point):
                    points.append(point.coords[0])
                else:
                    log.error("==2 : init point not contained in boundary! ( {} )".format(point))
                    exit() # TODO

        else:

            # case >= 3 points
            # arrange 'num' points equally spaced on a circle with 'dist' radius

            for i in range(0, num):
                randx = 0 # random.uniform(0.1, 1.0)
                randy = 0 # random.uniform(0.1, 1.0)
                point = Point([randx + center[0] + dist * math.cos(i*(2*math.pi/num)), randy + center[1] + dist * math.sin(i*(2*math.pi/num))])
                if self.boundary.contains(point):
                    points.append(point.coords[0])
                else:
                    log.error(">=3 : init point not contained in boundary! ( {} )".format(point))
                    exit() # TODO

            # TODO: if a gravitation vector is present, sort by attraction
            
        log.debug("generated init points: {}".format(points))

        return points


    def generate_subtree_names(self, tree):

        subtree_names = []

        # remove current node
        subtree = tree[1:]

        if len(subtree) == 0:
            return []

        if len(subtree) == 1 and "*" in subtree:
            return []

        subtree_without_cutouts = [x for x in subtree if not x == "*"]

        min_element = min(subtree_without_cutouts)
        indices = [i for i, x in enumerate(subtree) if x == min_element]

        for i in range(0, len(indices)):

            if i == len(indices)-1: # last element
                tree_candidate = subtree[indices[i]:]
            else:
                tree_candidate = subtree[indices[i]:indices[i+1]]

            subtree_names.append(tree_candidate)

        # TODO: subtree_names are currently not LHDS-ordered

        return subtree_names


    def calculate_weight_for_subtree(self, subtree):

        s = [int(x) for x in subtree]

        if len(s) <= 1:
            return 1
        else:
            return self.get_weight(subtree)


    def get_weight(self, subtree):

        if len(subtree) <= 1:
            return 1

        children = self.generate_subtree_names(subtree)
        weight = WEIGHTING_DEPTH_FACTOR + len(children)

        for c in children:
            weight += self.get_weight(c)

        return weight


    def is_leaf(self):

        if len(self.subtree_names) > 0:
            return False
        else:
            return True


    def get_leaves(self):

        if self.is_leaf():
            return [self]
        else:
            leaves = []
            for s in sorted(self.subtrees, reverse=True):
                leaves += s.get_leaves()

            return leaves


    def get_shape(self):

        if "*" not in self.tree:
            if LEAFS_AS_CIRLCES and self.is_leaf():

                if self.leaf_poly is None:
                    new_leaf, _, _ = self.get_largest_circle_in_poly(self.boundary, no_shrink=True)
                    if new_leaf is not None:
                        # if new_leaf.is_empty:
                        self.leaf_poly = new_leaf
                    else:
                        self.leaf_poly = self.boundary
                
                return self.leaf_poly

        boundary_poly = self.boundary

        if not self.depth == 0:
            # only smoothing, no shrinking
            boundary_poly = boundary_poly.buffer(-OVERSHRINK)
            boundary_poly = boundary_poly.buffer(OVERSHRINK)
        else: 
            # no corner smoothing for root node
            pass

        if type(boundary_poly) is MultiPolygon:
            log.warning("MultiPolygon")
            boundary_poly = self.get_largest_poly_of_multipolygon(boundary_poly)

        return boundary_poly


    def get_all_shapes(self):

        polys = []
        for tree in self.subtrees:
            polys = polys + tree.get_all_shapes()

        shape = self.get_shape()
        shape_non_overlapping = shape

        combined_polys = []
        for overlapping_poly, non_overlapping_poly, depth in polys:
            combined_polys.append(overlapping_poly)

        if len(combined_polys) > 0:
            combined_poly = ops.unary_union(combined_polys)
            shape_non_overlapping = shape.difference(combined_poly)

        polys.append((shape, shape_non_overlapping, self.depth))

        return polys


    def get_error(self):

        if len(self.subtrees) == 0:
            if self.old_error is None:
                return 0
            else:
                return np.average(np.abs(self.old_error))

        else:
            cumulative_error = []
            for subtree in self.subtrees:
                cumulative_error.append(subtree.get_error())

            return sum(cumulative_error)/len(cumulative_error)


    def _create_coord_pairs(self, coords):
        shifted = coords[1:] # + [coords[0]]
        return zip(coords[:-1], shifted)


    def reduce_polys(self):

        reduced = []

        for i in range(0, len(self.polys)):
            poly = self.polys[i]

            try:
                poly = self.boundary_shrinked.intersection(poly)

                if type(poly) is not Polygon:
                    poly = self.get_largest_poly_of_multipolygon(poly)

                poly_backup = Polygon(poly)

                poly = poly.buffer(-OVERSHRINK-self.config["boundary_shrink"])
                poly = poly.buffer(+OVERSHRINK)

                if poly.area < 1.0:
                    log.warning("poly backup required")
                    poly = poly_backup
                    poly = poly.buffer(-OVERSHRINK/2 - self.config["boundary_shrink"])
                    poly = poly.buffer(+OVERSHRINK/2)

                reduced.append(poly)

            except Exception as e:
                log.error("reduce polys failed: {}".format(e))

        return reduced

    def get_largest_poly_of_multipolygon(self, multipoly):

        polygons = []

        if type(multipoly) is Polygon:
            return multipoly

        elif type(multipoly) is GeometryCollection:
            for g in multipoly.geoms:
                if type(g) is Polygon:
                    polygons.append(g)

        elif type(multipoly) is MultiPolygon:
            polygons = multipoly.geoms

        polygons = sorted(polygons, key=attrgetter("area"), reverse=True)
        return self.get_largest_poly_of_multipolygon(polygons[0])


    def save(self, im, draw, transformation_matrix):

        polys = []
        polys.append(self.get_shape())

        if not self.finished:
            # node is not finalized and subtrees have not been created yet
            # just draw the current polys of the subtree (but with increased depth)
            polys = polys + self.reduce_polys()

        if self.depth == 0:
            draw.rectangle([(0, 0), im.size], fill=(0, 0, 0, 0))

        for i in range(0, len(polys)):

            poly = polys[i]

            if i == 0: # current node
                depth = self.depth
            else: # node not finalized, polys from potential subtrees
                depth = self.depth + 1

            if type(poly) is MultiPolygon:
                log.warning("MultiPolygon after reduction")
                poly = self.get_largest_poly_of_multipolygon(poly) 

            if not poly is None and len(list(poly.exterior.coords)) >= 3:

                coords = []
                for x, y in list(poly.exterior.coords):
                    coords.append(self._transform(x, y, transformation_matrix))

                # root node should be black
                if depth % 2 == 1:
                    fill = (255, 255, 255)
                else:
                    fill = (0, 0, 0)

                draw.polygon(coords, fill=fill)
            else:
                log.error("ERROR: invalid polygon. did not draw... {}".format(poly))

        for e in self.subtrees:
            e.save(im, draw, transformation_matrix)


    def _random_color(self, opacity=255):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), opacity)


    def _transform(self, x, y, transformation_matrix):
        xy = transformation_matrix * np.matrix([x, y, 1]).transpose()
        return (xy[0], xy[1])


    def _draw_poly_lines(self, draw, coords, transformation_matrix, width=1, fill=(0, 0, 0)):
        for pair in self._create_coord_pairs(coords):
            start = [
                self._transform(pair[0][0], pair[0][1], transformation_matrix)
            ]
            end = [
                self._transform(pair[1][0], pair[1][1], transformation_matrix)
            ]
            draw.line(start + end, width=width, fill=fill)


    def _save_debug(self, im, draw, transformation_matrix):

        # debug info
        # (only root should draw debug info to prevent garbled text)

        if self.depth == 0:

            draw.text(( 10, 10), "iteration:", font=font_large, fill=(0, 0, 0))
            draw.text((130, 10), "{:05}".format(self.iteration), font=font_large_bold, fill=(0, 0, 0))

            error = self.get_error()
            draw.text(( 10, 30), "error:", font=font_large, fill=(0, 0, 0))
            draw.text((130, 30), "{:9.7f}".format(error), font=font_large_bold, fill=(0, 0, 0))

            try:
                self.debug_errors
            except Exception as e:
                self.debug_errors = [[10, 60]]

            if self.iteration % 2 == 0:
                x = 10 + self.iteration
                y = 120 - (error * 500)       
                if y < 60:
                    y = 60
                self.debug_errors.append([x, y])

            r = 1
            for x, y in self.debug_errors:
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 0, 0))

            draw.line([(10, 120+5), (self.debug_errors[-1][0], 120+5)], width=1, fill=(0, 0, 0))

        # shape
        shape = self.get_shape()
        if shape is not None:

            if type(shape) is MultiPolygon:
                log.warning("drawing MultiPolygon")
                shape = self.get_largest_poly_of_multipolygon(shape)

            self._draw_poly_lines(draw, list(shape.exterior.coords), transformation_matrix, width=3, fill=(0, 0, 0))
            for hole in shape.interiors:
                self._draw_poly_lines(draw, hole.coords, transformation_matrix, width=3, fill=(0, 0, 0))

        # boundary
        self._draw_poly_lines(draw, list(self.boundary_shrinked.exterior.coords), transformation_matrix, width=1, fill=(255, 0, 0))
        for hole in self.boundary_shrinked.interiors:
            self._draw_poly_lines(draw, hole.coords, transformation_matrix, width=1, fill=(255, 0, 0))

        # circles (fill)
        if not self.finished:
            if self.centers is not None:
                for i in range(0, len(self.centers)):
                    c = list(self.centers[i])
                    # c = c[0]

                    x, y = self._transform(c[0], c[1], transformation_matrix)
                    r = self.weights[i] * self.local_scaling_factor[i] * transformation_matrix[0][0]
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0, 30))

        # polygons
        for i in range(0, len(self.polys)):
            poly = self.polys[i]
            color = self._random_color(opacity=30)
            # color = (0, 255, 0)

            if type(poly) is MultiPolygon:
                log.warning("drawing MultiPolygon")
                poly = self.get_largest_poly_of_multipolygon(poly)

            # raw cells
            if not self.finished and DEBUG_DRAW_VORONOI_CELLS:
                self._draw_poly_lines(draw, list(poly.exterior.coords), transformation_matrix, width=4, fill=color)

            # true polygons before shrinking
            try:
                poly = self.boundary_shrinked.intersection(poly)
                poly = poly.buffer(-self.config["boundary_shrink"])
                if type(shape) is MultiPolygon:
                    poly = self.get_largest_poly_of_multipolygon(poly)

                self._draw_poly_lines(draw, list(poly.exterior.coords), transformation_matrix, width=2, fill=(125, 125, 125))
                for hole in poly.interiors:
                    self._draw_poly_lines(draw, hole.coords, transformation_matrix, width=2, fill=(125, 125, 125))
            except Exception as e:
                log.error("drawing polygons failed: {}".format(e))

            # centroids
            if not self.finished:
                c = list(poly.centroid.coords)
                if not len(c) == 1:
                    log.error("no centroid found")
                    continue

                c = c[0]
                x, y = self._transform(c[0], c[1], transformation_matrix)

                draw.ellipse([x-7, y-7, x+7, y+7], fill=(0, 0, 0))

        # circles center
        if not self.finished and self.centers is not None:
            for i in range(0, len(self.centers)):
                c = list(self.centers[i])
                # c = c[0]
                x, y = self._transform(c[0], c[1], transformation_matrix)
                r = 7
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255))
                draw.text((x-2, y-4), "{}".format(i), font=font, fill=(0, 0, 0))


        # dummy centers
        if not self.finished:
            for i in range(0, len(self.dummy_centers)):
                c = list(self.dummy_centers[i])
                # c = c[0]
                x, y = self._transform(c[0], c[1], transformation_matrix)
                r = 1
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 0, 0))

        for e in self.subtrees:
            e._save_debug(im, draw, transformation_matrix)


    # taken from http://paulbourke.net/geometry/pointlineplane/
    def _point_line_dist(self, poly_coord_pairs, point):
        distances = []
        x0, y0 = point

        for ((x1, y1), (x2, y2)) in poly_coord_pairs:

            px = x2-x1
            py = y2-y1

            norm = px*px + py*py

            u =  ((x0 - x1) * px + (y0 - y1) * py) / float(norm)

            if u > 1:
                u = 1
            elif u < 0:
                u = 0

            x = x1 + u * px
            y = y1 + u * py

            dx = x - x0
            dy = y - y0

            dist = (dx*dx + dy*dy)**.5

            distances.append(dist)
        return distances


    def _min_dist(self, poly_coord_pairs, point):
        return min(self._point_line_dist(poly_coord_pairs, point))

 
    def _max_dist(self, poly_coord_pairs, point):
        return max(self._point_line_dist(poly_coord_pairs, point))


    def _shortest_distance_to_boundary(self, poly, point):
        return self._min_dist(self._create_coord_pairs(poly.exterior.coords), point)


    def _longest_distance_to_boundary(self, poly, point):
        return self._max_dist(self._create_coord_pairs(poly.exterior.coords), point)


    def get_largest_circle_in_poly(self, poly, no_shrink=False):

        poly_bounded = poly
        if not no_shrink:
            poly_bounded = poly_bounded.buffer(-self.config["boundary_shrink"])
        poly_bounded = poly_bounded.simplify(LARGEST_CIRCLE_PRECISION)

        if not poly_bounded.is_valid or poly_bounded.is_empty:
            log.error("could not get largest circle, poly not within boundary")
            return None, None, None

        minx, miny, maxx, maxy = poly_bounded.bounds

        bestc = None
        bestr = -1

        coord_pairs = list(self._create_coord_pairs(poly_bounded.exterior.coords))
        for hole in poly_bounded.interiors:
            coord_pairs += list(self._create_coord_pairs(hole.coords))

        for x0 in np.linspace(minx, maxx, int((maxx-minx)/LARGEST_CIRCLE_PRECISION)):
            for y0 in np.linspace(miny, maxy, int((maxy-miny)/LARGEST_CIRCLE_PRECISION)):

                p = Point([x0, y0])
                if not poly_bounded.contains(p): 
                    continue

                distance = self._min_dist(coord_pairs, [x0, y0])

                if distance > bestr + 0.1:
                    bestc = [x0, y0]
                    bestr = distance
                        
                    # log.debug("new best point: {} {}".format(bestc, bestr))

        if not bestc is None:
            log.debug("circle: best point: {} {}".format(bestc, bestr))
            return Point(bestc).buffer(bestr), bestc, bestr
        else:
            return None, None, None


    def validate(self):

        validation_result = True

        if not self.boundary.is_valid:
            log.error("node {} has invalid boundary".format(self))
            validation_result = False

        if self.boundary.is_empty:
            log.error("node {} has empty boundary".format(self))
            validation_result = False

        if self.boundary.area < MIN_FEATURE_SIZE:
            log.error("node {} has area below MIN_FEATURE_SIZE: {}".format(self, self.boundary.area))
            validation_result = False

        for s in self.subtrees:
            valid = s.validate()
            if not valid:
                validation_result = False

        return validation_result


    def finalize(self):

        if self.finished:
            log.error("already finished")
            return

        self.finished = True

        log.info("finalize tree {}. generating {} subtree(s)".format("".join(self.tree), len(self.subtree_names) + len(self.fixed_subtrees)))

        for i in range(0, len(self.polys)):
            subtree_boundary = self.polys[i]

            subtree_boundary = subtree_boundary.intersection(self.boundary_shrinked)
            subtree_boundary = subtree_boundary.buffer(-self.config["boundary_shrink"])

            if type(subtree_boundary) is MultiPolygon:
                log.warning("finalized boundary is MultiPolygon")
                subtree_boundary = self.get_largest_poly_of_multipolygon(subtree_boundary)

            if subtree_boundary.is_empty:
                log.error("failed to pass empty boundary to subtree (poly number: {})".format(i))
                log.error("probably the tree is to large/imbalanced and no good fit could be found given the minimum boundary distances.")
                exit(ERROR_EMPTY_BOUNDARY_SUBTREE)

            self.subtrees.append(Subtree(self.subtree_names[i], subtree_boundary, config, gravity_vector=self.gravity_vector, cutouts=self.cutout_list, init_points=self.local_init_points, depth=self.depth+1))

        for i in range(0, len(self.cutouts)):
            cutout_boundary = self.cutouts[i]

            cutout_boundary = cutout_boundary.buffer(self.config["boundary_shrink"]*2)
            cutout_boundary = self.boundary_shrinked.intersection(cutout_boundary)

            if type(cutout_boundary) is MultiPolygon:
                log.warning("finalized cutout is MultiPolygon")
                cutout_boundary = self.get_largest_poly_of_multipolygon(cutout_boundary)

            if cutout_boundary.is_empty:
                log.error("failed to pass empty boundary to fixed subtree (cutout number: {})".format(i)) 
                log.error("probably the tree is to large/imbalanced and no good fit could be found given the cutouts")
                exit(ERROR_EMPTY_BOUNDARY_SUBTREE)

            self.subtrees.append(Subtree(self.fixed_subtrees[i], cutout_boundary, self.config, depth=self.depth+1))

    def _get_distance_to_centers(self, centers, point, weight, index_to_ignore=None):

        # compute distances among centers
        distance = np.ma.array(centers.copy()) # compare with all centers, incl. dummy centers

        distance = distance - point
        distance = np.power(distance, 2)
        distance = np.sum(distance, axis=1)
        distance = np.sqrt(distance)

        distance = np.subtract(distance, weight)

        mask = np.zeros((centers.shape[0], 1), dtype=np.bool)

        if index_to_ignore is not None:
            mask[index_to_ignore] = True

        distance.mask = mask 

        return distance


    def get_seedmarker_descriptor(self):

        desc = ""

        desc += self.get_lhds_name()

        for l in self.get_leaves():
            desc += "|"
            _, bestc, bestr = l.get_largest_circle_in_poly(l.boundary)
            if not None in (bestc, bestr):
                desc += "{:.2f}:{:.2f}:{:.2f}".format(*bestc, bestr)
            else:
                desc += "_"

        return desc


    def run(self):

        self.iteration += 1

        if not self.finished:
            log.debug("tree {} | iteration: {}".format("".join(self.tree), self.iteration))
    
        if not self.finished and len(self.subtree_names) == 0:
            self.finalize()
            return False

        if not self.finished and self.iteration == MAX_ITERATIONS:
            self.finalize()
            return False

        if self.finished:

            done = True

            for e in self.subtrees:
                subtree_done = e.run()

                if not subtree_done:
                    done = False

            return done

        self.polys = []

        num_elements = self.centers.shape[0] + len(self.dummy_centers) 

        centers = self.centers
        weights = np.multiply(self.weights, self.local_scaling_factor)

        if len(self.dummy_centers) > 0:
            centers = np.append(centers, self.dummy_centers, axis=0)
            weights = np.append(weights, np.full([len(self.dummy_centers),], fill_value=0.01), axis=0)

        # two/three or more elements, compute power diagram
        if num_elements >= 2:

            voronoi_cell_map = powerwrapper.get_power_diagram(centers, weights, boundary=self.boundary.bounds)

            for i, segment_list in voronoi_cell_map.items():

                # do not process dummy centers
                if i >= len(self.centers):
                    continue

                points = []

                for a in range(0, 10): # attempts
                    for (edge, (A, U, tmin, tmax)) in segment_list:

                        dist = 400 - 40*a # TODO: get rid of the magic number. Bounding box diagonale should guarantee sufficient long bisectors

                        if tmax is None:
                            tmax = dist
                        if tmin is None:
                            tmin = -dist

                        points.append((A + tmin * U).tolist())
                        points.append((A + tmax * U).tolist())

                    if len(points) >= 3:
                        p = MultiPoint(points).convex_hull

                        if type(p) is MultiPolygon:
                            log.warning("MultiPolygon")
                            p = self.get_largest_poly_of_multipolygon(p) 

                        if p.is_valid and not p.is_empty:
                            self.polys.append(p)
                            break
                        else:
                            p = p.buffer(0.00001)
                            if p.is_valid and not p.is_empty:
                                log.error("polygon {} invalid but fixed. (attempt: {})".format(i, a))

                                if type(p) is MultiPolygon:
                                    log.warning("MultiPolygon")
                                    p = self.get_largest_poly_of_multipolygon(p) 

                                self.polys.append(p)
                                break
                            else:
                                log.error("polygon {} invalid. (attempt: {})".format(i, a))
                                log.error(p)
                    else:
                        log.error("not enough points to create polygon")

        # only one element, do nothing
        elif self.centers.shape[0] == 1:
            self.polys.append(Polygon(self.boundary))

        else:
            log.error("ERROR: empty tree")
            exit(ERROR_EMPTY_TREE)

        # area calculation
        circle_radius_total = np.sum(self.weights)
        poly_areas_total = self.boundary.area
        polys_bounded = []

        # move to centroid
        for i in range(0, len(self.polys)):

            poly = self.polys[i]
            try:
                poly = self.boundary.intersection(poly)

                for c in self.cutouts:
                    poly = poly.difference(c)

                if type(poly) is MultiPolygon:
                    log.warning("MultiPolygon")
                    poly = self.get_largest_poly_of_multipolygon(poly)

                polys_bounded.append(poly)
            except Exception as e:
                log.warning("error at poly {}: {}".format(i, e))

            # TODO: move centers if collision occurs

            # aggregate movement vectors
            vectors = []

            # movement towards centroid
            c = list(poly.centroid.coords)

            if not len(c) == 1:
                log.warning("ERROR: no centroid found")
                continue

            x = c[0][0] - self.centers[i][0]
            y = c[0][1] - self.centers[i][1] 

            vectors.append([x * CENTROID_CHANGE_RATE, y * CENTROID_CHANGE_RATE])

            # gravity
            if self.gravity_vector is not None:
                if self.gravity[i] == 1:
                    vectors.append([self.gravity_vector[0] * GRAVITY_PULL_RATE, self.gravity_vector[1] * GRAVITY_PULL_RATE])
                
            # combined movement vector
            cvec = [0, 0]
            for v in vectors:
                cvec[0] += v[0]
                cvec[1] += v[1] 

            # print("combined movement vector {}: {:5.2f} {:5.2f}".format(i, *cvec))
            new_center = [self.centers[i][0] + cvec[0], self.centers[i][1] + cvec[1]]

            move = True

            # for cutout in self.cutouts:
            #     new_center_circle = Point(new_center).buffer(self.weights[i] * self.local_scaling_factor[i])
            #     if cutout.intersects(new_center_circle):
            #         move = False
            #         log.warning("CUTOUT COLLISION")

            if CHECK_FOR_COLLISION:

                # new_distance_to_boundary = self._shortest_distance_to_boundary(self.boundary, new_center)

                # if new_distance_to_boundary < 0.1:
                #     move = False

                #     min_distance_to_boundary = self._shortest_distance_to_boundary(self.boundary, self.centers[i])
                #     old_distance = min_distance_to_boundary

                #     if new_distance_to_boundary > old_distance + 0.1:
                #         move = True

                if not self.boundary_with_cutouts.contains(Point(new_center)):
                    move = False

                else:
                    min_distance_to_centers = np.min(self._get_distance_to_centers(centers, self.centers[i], self.weights[i] * self.local_scaling_factor[i], index_to_ignore=i))
                    new_distance_to_centers = np.min(self._get_distance_to_centers(centers, new_center, self.weights[i] * self.local_scaling_factor[i], index_to_ignore=i))

                    if new_distance_to_centers < 0.1:
                        move = False
                        old_distance = min_distance_to_centers

                        if new_distance_to_centers > old_distance + 0.1:
                            move = True
            
            if move:
                self.centers[i] = new_center
            else:
                log.warning("COLLISION")
                # print("new distance: {}".format(new_distance))
                # print("old distance: {}".format(old_distance))
                # self.centers[i] = new_center

            # ----------------------------------------------------------------------

            # change local scaling factor
            
            soll = self.weights[i] / circle_radius_total
            ist = poly.area / poly_areas_total
            scaling_error = soll-ist

            local_scaling_factor_change = scaling_error * SCALING_FACTOR_CHANGE_RATE
            new_local_scaling_factor = self.local_scaling_factor[i] + local_scaling_factor_change
            if new_local_scaling_factor < 1:
                new_local_scaling_factor = 1

            new_distance_to_centers = np.min(self._get_distance_to_centers(centers, new_center, self.weights[i] * new_local_scaling_factor, index_to_ignore=i))

            if new_distance_to_centers < 1.0: # and local_scaling_factor_change > 0:
                pass  
            else:
                if self.iteration >= MIN_ITERATIONS_BEFORE_SCALING:
                    self.local_scaling_factor[i] = new_local_scaling_factor
    
            distance_moved = math.sqrt(cvec[0] ** 2 + cvec[1] ** 2)
            total_error = distance_moved * 0.5 + scaling_error * 0.5
            self.new_error[i] = total_error

        mean_error_change = np.mean(np.abs(self.old_error - self.new_error))
        self.old_error = self.new_error.copy()

        # log.debug("mean error change: {:12.8}".format(mean_error_change))

        if mean_error_change < MIN_ERROR and self.iteration >= MIN_ITERATIONS:
            log.debug("min error threshold cut, tree finished (mean error change: {})".format(mean_error_change))
            self.finalize()
            return False # not done (subtress must be iterated)

        return False # not done


def create_image(transparent=False):

    image_type = "RGB"
    if transparent:
        image_type = "RGBA"

    im = Image.new(image_type, OUTPUT_IMAGE_DIMENSIONS, (255,255,255,255))
    draw = ImageDraw.Draw(im, "RGBA")

    return im, draw


if __name__ == "__main__":

    start = datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input", 
        type=str,
        help="the DXF file that describes the outline of the marker"
    )

    parser.add_argument(
        "graph", 
        type=str,
        help="the left heavy depth sequence of the topological graph"
    )

    parser.add_argument(
        "-o",
        "--output", 
        type=str,
        default="output.dxf",
        help="the DXF file that contains the marker. default: output.dxf"
    )

    parser.add_argument(
        "--output-image", 
        type=str,
        default=None,
        help="the PNG file that contains the marker. default: None"
    )

    parser.add_argument(
        "--debug-directory", 
        type=str,
        default="seedmarker_debug",
        help="directory for debug image output"
    )

    parser.add_argument(
        "--line-width", 
        type=float,
        default=LINE_WIDTH,
        help="distance between two topological structures"
    )

    parser.add_argument(
        "--reactivision", 
        type=float,
        default=None,
        help="ReacTIVision legacy mode [no cutouts possible]. Orientation vector required (0deg = north, clockwise)"
    )

    parser.add_argument(
        "--hexagon-fill-radius", 
        type=float,
        default=None,
        help="creates hexagon shapes filling every second boundary. radius need to be lesser than LINE WIDTH"
    )

    parser.add_argument(
        "--verbose", 
        help="increase output verbosity",
        action="store_true"
    )

    parser.add_argument(
        "--precise", 
        help="increase output precision",
        action="store_true"
    )

    args = parser.parse_args()

    config = {}

    config["line_width"] = args.line_width
    config["boundary_shrink"] = config["line_width"] / 2

    for key in config.keys():
        log.info("{:20}: {}".format(key, config[key]))

    # create logger
    logger = log.getLogger() 
    logger.setLevel(log.DEBUG)

    log.basicConfig(level=log.DEBUG)
    #                     format="%(asctime)s | %(levelname)-7s | %(message)s",
    #                     datefmt='%m-%d %H:%M',
    # )



    # subloggers
    ezdxf_logger = log.getLogger("ezdxf").setLevel(log.WARN)

    # formatter = log.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    # consoleHandler = log.StreamHandler()
    # consoleHandler.setLevel(log.DEBUG)
    # consoleHandler.setFormatter(formatter)
    # logger.addHandler(consoleHandler)

    root = log.getLogger()
    root_handler = root.handlers[0]
    formatter = log.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    root_handler.setFormatter(formatter)

    if args.precise:
        pass # TODO

    if not os.path.exists(args.debug_directory):
        os.mkdir(args.debug_directory)
    if not os.path.exists(args.debug_directory + "_clear"):
        os.mkdir(args.debug_directory + "_clear")

    log.info("Reading DXF: {}".format(args.input))

    boundaries, artificial_init_points = dxfread.read(args.input)
    boundaries = sorted(boundaries, key=attrgetter("area"), reverse=True)

    if len(boundaries) == 0:
        log.error("unable to dervice boundary from DXF file {}".format(args.input))
        log.error("DXF file may use incompatible arcs or non polyline elements.")
        exit(ERROR_ILLEGAL_INPUT)

    # largest polygon is assumed to be main, smaller ones cutout
    boundary = boundaries[0]
    boundary = boundary.simplify(SIMPLIFICATION_MAX_ERROR)
    cutouts = boundaries[1:]

    if len(cutouts) > 0:
        log.info("loaded {} cutouts:".format(len(cutouts)))
  
    if len(artificial_init_points) > 0:
        log.info("loaded {} init points:".format(len(artificial_init_points)))

    if args.reactivision is not None and len(cutouts) > 0:
        log.error("ReaTIVision orientation vector is supplied. No cutouts supported (found {} cutouts in DXF: {}".format(len(cutouts), args.input))
        exit(ERROR_ILLEGAL_PARAM)

    """

    Problem:

    Cutout is polygon. So we need the smallest circle around the poly (i.e. center and radius)
    Then we need to guarantee that this will never be cut by the other subtrees
        Problem:
            weights are not mm/cm units but only relative scaling factors

    Option A:
        cutout is a real cutout from the boundary shape
            -> results in a MultiPolygon/Polygon with hole == harder to handle
            Problem: what happens when a polygon from a lower-layer subtree engulfs the hole? -> cutout would move to the wrong subtree

    Option B: 
        cutout is a subtree with a fixed boundary, can not be altered or moved but is not a voronoi cell
            Problem: power diagram won't know how to handle this

    Option C:
        cutout is a subtree with a non-fixed boundary, just a regular voronoi cell but won't be moved and weights guarantee size
            Problem: weights don't correspond to 2D units, they are just a relative weighting

    Option C-2:
        cutout is introducing dummy voronoi cells with nearly-zero weights placed along the cutouts polygon. 
        These dummy cells are then ignored and provide an empty space guaranteed to be at least the size of the cutout
            Problem: may waste a lot of space


    """

    cutout_list = []
    for c in cutouts:
        depth = 1
        cutout_list.append([depth, c])

    left_heavy_depth_tree_str = args.graph
    left_heavy_depth_tree = []

    for char in left_heavy_depth_tree_str:
        left_heavy_depth_tree.append(char) #int(char))

    # remove ReacTIVisions w/b prefix from graph names if present
    if left_heavy_depth_tree[0] in ["w", "b"] and args.reactivision:
        left_heavy_depth_tree = left_heavy_depth_tree[1:]

        # increment every element
        for i in range(0, len(left_heavy_depth_tree)):
            item = int(left_heavy_depth_tree[i])
            left_heavy_depth_tree[i] = str(item+1)

        left_heavy_depth_tree = ["0"] + left_heavy_depth_tree

    gravity_vector = None
    if args.reactivision is not None:
        gravity_vector = [math.cos(math.radians(args.reactivision)), math.sin(math.radians(args.reactivision))]
        log.debug("gravity  : {:4.2f} {:4.2f}".format(*gravity_vector))

    root = Subtree(left_heavy_depth_tree, boundary, config, gravity_vector=gravity_vector, cutouts=cutout_list, init_points=artificial_init_points)

    # compute transformation matrix for image outputs

    minx, miny, maxx, maxy = boundary.bounds
    long_side_boundary = max(maxx-minx, maxy-miny)
    long_side_image = max(*OUTPUT_IMAGE_DIMENSIONS) - OUTPUT_PADDING*2
    scale = long_side_image / long_side_boundary
    if FIXED_OUTPUT_SCALE_FACTOR is not None:
        scale = FIXED_OUTPUT_SCALE_FACTOR
    image_center = [OUTPUT_IMAGE_DIMENSIONS[0]/2.0, OUTPUT_IMAGE_DIMENSIONS[1]/2.0]
    object_center = [minx+(maxx-minx)/2.0, miny+(maxy-miny)/2.0] 

    transformation_matrix = np.identity(3)
    transformation_matrix[0][2] = image_center[0] - object_center[0] * scale
    transformation_matrix[1][2] = image_center[1] - object_center[1] * scale
    transformation_matrix[0][0] = scale
    transformation_matrix[1][1] = scale

    # main loop
    while True: 
        finished = root.run()

        if root.iteration % EXPORT_DEBUG_IMAGE_SKIP == 0:

            if EXPORT_DEBUG_CLEAR_IMAGE:
                im, draw = create_image(transparent=True)
                root.save(im, draw, transformation_matrix)
                im.save(os.path.join(args.debug_directory + "_clear", OUTPUT_NAME.format(root.iteration)), "PNG")

            if EXPORT_DEBUG_IMAGE:
                im, draw = create_image()
                root._save_debug(im, draw, transformation_matrix)
                im.save(os.path.join(args.debug_directory, OUTPUT_NAME.format(root.iteration)), "PNG")

        if finished:
            break
    
    if EXPORT_DEBUG_CLEAR_IMAGE:
        im, draw = create_image(transparent=True)
        root.save(im, draw, transformation_matrix)
        im.save(os.path.join(args.debug_directory + "_clear", OUTPUT_NAME.format(root.iteration+1)), "PNG")

    if EXPORT_DEBUG_IMAGE:
        im, draw = create_image()
        root._save_debug(im, draw, transformation_matrix)
        im.save(os.path.join(args.debug_directory, OUTPUT_NAME.format(root.iteration+1)), "PNG")

    if args.output_image is not None:
        im, draw = create_image()
        root.save(im, draw, transformation_matrix)
        im.save(args.output_image, "PNG")

    writer = WriterSimpleDXF(args.output)

    for poly, non_overlap_poly, depth in root.get_all_shapes():

        if args.hexagon_fill_radius is not None:
            poly_simplified = non_overlap_poly.simplify(SIMPLIFICATION_MAX_ERROR)

            if depth == 0:
                poly_simplified = poly_simplified.buffer(-args.hexagon_fill_radius)

            if depth % 2 == 1:
                continue
            
            DISTANCE = 0.3

            minx, miny, maxx, maxy = poly_simplified.bounds
            width = maxx-minx
            height = maxy-miny

            # Circles

            # num_x = width / (args.hexagon_fill_radius*2 + DISTANCE)
            # num_y = height / (args.hexagon_fill_radius*2 + DISTANCE)

            # for i in range(0, int(num_x)):
            #     for j in range(0, int(num_y)):

            #         x = minx + (args.hexagon_fill_radius*2 + DISTANCE) / 2 + (args.hexagon_fill_radius*2 + DISTANCE) * i 
            #         y = miny + (args.hexagon_fill_radius*2 + DISTANCE) / 2 + (args.hexagon_fill_radius*2 + DISTANCE) * j

            #         h = Point([x, y]).buffer(args.hexagon_fill_radius)
            #         h = poly_simplified.intersection(h)

            #         if h.is_empty:
            #             continue

            #         writer.add_path(list(h.exterior.coords), close_path=True)
            #         for hole in h.interiors:
            #             writer.add_path(list(hole.coords), close_path=True)

            # Hexagons

            w = math.sqrt(3) * (args.hexagon_fill_radius + DISTANCE)
            h = 2 * (args.hexagon_fill_radius + DISTANCE)
            num_x = width / w + 1
            num_y = height / (0.75*h) + 1

            thirdPi = math.pi / 3
            angles = [0, 1 * thirdPi, 2 * thirdPi, 3 * thirdPi, 4 * thirdPi, 5 * thirdPi]
            hexagon_points = []
            for angle in angles:
                x = math.sin(angle) * args.hexagon_fill_radius
                y = -math.cos(angle) * args.hexagon_fill_radius
                hexagon_points.append([x, y])
            hexagon = Polygon(hexagon_points)

            for i in range(0, int(num_y)):
                for j in range(0, int(num_x)):

                    x = minx + w * j
                    y = miny + (0.75*h) * i

                    if i % 2 == 0:
                        x += 0.5 * w

                    p = affinity.translate(hexagon, xoff=x, yoff=y)
                    
                    p_i = poly_simplified.intersection(p)

                    if p_i.is_empty:
                        continue

                    if p_i.area < hexagon.area * 0.6:
                        continue

                    # if not poly_simplified.contains(p):
                    #     continue

                    writer.add_path(list(p.exterior.coords), close_path=True)
                    for hole in p.interiors:
                        writer.add_path(list(hole.coords), close_path=True)

        else:
            poly_simplified = poly.simplify(SIMPLIFICATION_MAX_ERROR)
            writer.add_path(list(poly_simplified.exterior.coords), close_path=True)

            for hole in poly_simplified.interiors:
                writer.add_path(list(hole.coords), close_path=True)

    # write boundary separately if hexagon substitution is done
    if args.hexagon_fill_radius is not None:        
        p = root.boundary
        writer.add_path(list(p.exterior.coords), close_path=True)
        for hole in p.interiors:
            writer.add_path(list(hole.coords), close_path=True)

    writer.save()

    log.info("done. took: {:.2f}s".format((datetime.now()-start).total_seconds()))

    valid = root.validate()
    if not valid:
        log.error("failed validation")
        exit(ERROR_FAILED_VALIDATION)

    seedmarker_descriptor = root.get_seedmarker_descriptor()
    log.info("SEEDMARKER DESCRIPTOR: {}".format(seedmarker_descriptor))

    with open(args.output[:-3] + "txt", "w") as f:
        f.write("{:40}: {}\n".format("outline file", args.input))
        f.write("{:40}: {}\n".format("graph", args.graph))
        if args.reactivision is not None:
            f.write("{:40}: {}\n".format("reactivision orientation vector", args.reactivision))
        f.write("{:40}: {}\n".format("seedmarker descriptor", seedmarker_descriptor))
       