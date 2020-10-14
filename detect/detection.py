import cv2
import numpy as np
import json
import time
import argparse
import os

MIN_DEPTH = 2
MAX_DEPTH = 6
# MIN_LEAVES = 2
MAX_WIDTH = 10

class Node(object):

    def __init__(self, contour_index, contour, parent=None):
        self.contour_index = contour_index
        self.contour = contour
        self.parent = parent
        self.children = []
        self.weight = -1
        self.is_unique = None

        self.rvec = None
        self.tvec = None

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children_list):
        self.children += children_list 

    def remove_child(self, child):
        self.children.remove(child)

    def get_children(self):
        return self.children

    def has_children(self):
        if len(self.children) > 0:
            return True
        else:
            return False

    def get_weight(self):
        if len(self.children) == 0:
            return 1
        else:
            weight = 1
            for c in self.children:
                weight += c.get_weight()
            return weight

    def get_leaves(self):

        if len(self.children) == 0:
            return [self]
        else:
            leaves = []
            for c in self.children: 
                leaves += c.get_leaves()

        return leaves

    def remove_duplicates_recursively(self):
        grandchildren = []
        remove_children = []
        if self.has_children():
            for c in self.get_children():
                cont = c.contour 
                if len(cont) > 1:
                    d = cont[0][0][0] - cont[1][0][0]
                    if d < 0:
                        grandchildren += c.get_children()
                        remove_children.append(c)

            for rc in remove_children:
                self.remove_child(rc)

            self.add_children(grandchildren) 

        for c in self.get_children():
            c.remove_duplicates_recursively()

    def sort_tree(self):
        for c in self.children:
            c.sort_tree()

        self.children = sorted(self.children, reverse=True)

    def __eq__(self, other): 
        if not isinstance(other, Node):
            return NotImplemented

        return self.contour_index == other.contour_index

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        return int(self.get_lhds_name()) <= int(other.get_lhds_name())


    def __repr__(self):
        return "node [{}]".format(self.contour_index) # contour index is the only thing which is truly unique

    def get_max_depth(self, depth=0):
        depths = [depth]
        for c in self.children:
            depths.append(c.get_max_depth(depth=depth+1))

        return max(depths)

    def get_max_width(self):
        widths = [len(self.children)]
        for c in self.children:
            widths.append(c.get_max_width())

        return max(widths)

    def get_all_contour_indices(self):
        res = []
        for c in self.children:
            res += c.get_all_contour_indices()

        return res + [self.contour_index]

    def get_lhds_name(self, depth=0):

        name = str(depth)
        children_sorted = self.get_children() #sorted(self.get_children(), reverse=True)
        for c in children_sorted:
            name += c.get_lhds_name(depth+1)

        return name

    def compute_uniques(self, unique):

        self.is_unique = unique

        if not unique:
            for c in self.children:
                c.compute_uniques(False)
        else:
            children_names_hashed = {}
            for i in range(0, len(self.children)):
                c = self.children[i]
                name = c.get_lhds_name()
                if name not in children_names_hashed:
                    children_names_hashed[name] = [i]
                else:
                    children_names_hashed[name] += [i]

            for indices in children_names_hashed.values():
                if len(indices) == 1:
                    self.children[indices[0]].compute_uniques(True)
                else:
                    for i in indices:
                        self.children[i].compute_uniques(False)

    def find_subtree_candidates(self):

        results = []

        if len(self.children) == 1:
            grandchildren = self.children[0].children
            if len(grandchildren) == 1:
                if len(grandchildren[0].children) > 1:
                    if len(self.get_leaves()) >= MIN_LEAVES:
                        results.append(self)

        if len(results) == 0:
            for c in self.children:
                results += c.find_subtree_candidates()

        return results

    def find_subtree(self, lhds_name):

        results = []

        if self.get_lhds_name() == lhds_name:
            results.append(self)
            return results

        for c in self.children:
            results += c.find_subtree(lhds_name)

        return results

    def add_pose(self, rvec, tvec):
        self.rvec = rvec
        self.tvec = tvec


def parse_contour_hierarchy(contours, hierarchy):

    # hierarchy      0     1     2      3
    # contour index [next, prev, child, parent]

    if contours is None or hierarchy is None:
        return []

    top_level_nodes = []
    indices_visited_globally = []

    for i in range(0, len(hierarchy[0])):
        node_list, indices_visited = traverse_graph(contours, hierarchy, i, None, indices_visited_globally)
        indices_visited_globally = indices_visited
        top_level_nodes += node_list

        if len(indices_visited) >= len(hierarchy[0]):
            break

    # guarantees LHD order for all trees
    for tln in top_level_nodes:
        tln.sort_tree()

    return top_level_nodes

def traverse_graph(contours, hierarchy, index, parent, indices_visited, depth=0):

    if depth > 1000:
        return [], indices_visited

    line = hierarchy[0][index]

    next_e  = line[0]
    prev_e  = line[1]
    child   = line[2]
    parent  = line[3]

    if index in indices_visited:
        return [], indices_visited
    else:
        indices_visited.append(index)

    node_list = []
    node = Node(index, contours[index], parent=parent)
    node_list.append(node)

    if child > -1:
        children_nodes, indices_visited = traverse_graph(contours, hierarchy, child, node, indices_visited, depth=depth+1) 
        node.add_children(children_nodes)

    if next_e > -1:
        sibling_nodes, indices_visited = traverse_graph(contours, hierarchy, next_e, parent, indices_visited, depth=depth+1)
        node_list += sibling_nodes

    return node_list, indices_visited

def match_leaves(descriptor, detected_tree):

    print(detected_tree.get_lhds_name())
    detected_tree.compute_uniques(True)

    leaves = detected_tree.get_leaves()
    print("leaves ({}): {}".format(len(leaves), leaves))
    unique_leaves = [x for x in leaves if x.is_unique]
    print("unique leaves ({}): {}".format(len(unique_leaves), unique_leaves))

    tree_name = descriptor.split("|")[0]
    descriptor_circles = descriptor.split("|")[1:]

    print("descriptor leaves ({})".format(len(descriptor_circles)))
    print(descriptor_circles)

    circles_pos = []
    leaves_pos = []
    for i in range(0, len(descriptor_circles)):

        detected_leaf = leaves[i]

        if not detected_leaf.is_unique:
            continue

        # just use the treename and the ellipses' center
        # for matching. completely ignore the radius
        # and ellipse for now and the fact that the
        # projection makes our ellipsis centers unreliable

        if len(detected_leaf.contour) >= 5: # min amount of points to fit ellipse

            rotated_rect = cv2.fitEllipse(detected_leaf.contour)

            width = rotated_rect[0][0] - rotated_rect[1][0]
            height = rotated_rect[0][1] - rotated_rect[1][1]

            center = [
                rotated_rect[0][0], # + width / 2.0,
                rotated_rect[0][1]  # + height / 2.0
            ]
            leaves_pos.append([*center]) #, (width+height)/2.0])

        else:
            continue

        circles_pos.append([float(x) for x in descriptor_circles[i].split(":")[0:2]])

        # else:
        #     M = cv2.moments(detected_leaf.contour)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])

        #     leaves_pos.append([cX, cY])

    return (circles_pos, leaves_pos)


def process(image_cap, lhds, descriptor):

    # scale_percent = 50 # percent of original size
    # width = int(image_cap.shape[1] * scale_percent / 100)
    # height = int(image_cap.shape[0] * scale_percent / 100)
    # dim = (width, height) 
    # image_cap = cv2.resize(image_cap, dim, interpolation = cv2.INTER_AREA)

    # dim = (1080, 1920) #(1920, 1080) 
    # image_cap = cv2.resize(image_cap, dim, interpolation = cv2.INTER_AREA)

    if len(image_cap.shape) == 2:
        gray = image_cap
    else:
        gray = cv2.cvtColor(image_cap, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # edged = cv2.Canny(blurred, 50, 150)
    _, edged = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    edges_for_contour = edged.copy()

    # find contours in the edge map
    contours, hierarchy = cv2.findContours(edges_for_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    top_level_nodes = parse_contour_hierarchy(contours, hierarchy)

    if args.draw_output:
        # output_img = edged.copy()
        # output_img = gray.copy()
        output_img = image_cap
        cv2.drawContours(output_img, contours, -1, (0, 255, 0), 1)

    total_matches = []

    for tln in top_level_nodes:

        # if tln.get_max_depth() >= MIN_DEPTH:
        #     print("TLN: {}".format(tln.get_lhds_name()))

        matches = []

        if descriptor is not None:
            desc = descriptor.split("|")[0]
            # desc = "".join([str(int(x)-1) for x in desc[1:]])
            matches = tln.find_subtree(desc)
        else:
            total_matches.append(tln)

        for m in matches:

            total_matches.append(m)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.75
            fontColor              = (0, 255, 0)
            lineType               = 2

            ops, ips = match_leaves(descriptor, m)

            if len(ips) >= 4:

                # calibration data is using m, descriptor is mm
                object_points = np.zeros([len(ops), 3])
                object_points[:, 0:2] = np.multiply(ops, 0.001)
                image_points = np.matrix(ips)

                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                if success:
                    m.add_pose(rvec, tvec)
                else:
                    m.add_pose(None, None)

                if args.draw_output:

                    for i in range(0, len(ips)):
                        op = ops[i]
                        ip = ips[i]

                        cv2.circle(output_img, (int(ip[0]), int(ip[1])), 5, (255, 0, 0), -1)

                        # cv2.putText(
                        #     output_img,
                        #     "{:6.1f}, {:6.1f}".format(ip[0], ip[1]), 
                        #     (int(ip[0]), int(ip[1])), 
                        #     font, 
                        #     fontScale,
                        #     fontColor,
                        #     lineType)

                        # cv2.putText(
                        #     output_img,
                        #     "{:6.1f}, {:6.1f}".format(op[0], op[1]), 
                        #     (int(ip[0]), int(ip[1])+22), 
                        #     font, 
                        #     fontScale,
                        #     fontColor,
                        #     lineType)

                    cv2.aruco.drawAxis(output_img, camera_matrix, dist_coeffs, rvec, tvec, 10)
            else:
                m.add_pose(None, None)
    
    if args.draw_output:
        dim = (int(1920/2), int(1080/2)) #(1920, 1080) 
        # output_img = cv2.resize(output_img, dim, interpolation = cv2.INTER_AREA)

        while(True):
            cv2.imshow('Frame', output_img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    return total_matches


def validate_node(node):

    width = node.get_max_width()
    depth = node.get_max_depth()

    if width > MAX_WIDTH:
        return []

    if depth > MAX_DEPTH:
        return []

    if depth < MIN_DEPTH:
        return []

    valid_nodes = [(node.get_lhds_name(), width, depth)]

    for c in node.children:
        valid_nodes += validate_node(c)

    return valid_nodes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-image", 
        type=str,
        default=None,
        help="input image"
    )

    parser.add_argument(
        "--input-image-directory", 
        type=str,
        default=None,
        help="input image directory"
    )

    parser.add_argument(
        "--input-video", 
        type=str,
        default=None,
        help="input video"
    )

    parser.add_argument(
        "--input-device", 
        type=str,
        default=None,
        help="input device"
    )

    parser.add_argument(
        "--calibration-data", 
        type=str,
        default=None,
        help="calibration data"
    )

    parser.add_argument(
        "--descriptor", 
        type=str,
        default=None,
        help="descriptor"
    )

    parser.add_argument(
        "--draw-output", 
        help="draw the detected graphs on the output image",
        action="store_true"
    )

    parser.add_argument(
        "--output-file", 
        default=None,
        help="pose output to file"
    )

    parser.add_argument(
        "--verbose", 
        help="verbose graph info output",
        action="store_true"
    )

    args = parser.parse_args()

    camera_matrix = None
    dist_coeffs = np.zeros([1, 5])
    if args.calibration_data is not None:
        with open(args.calibration_data, "r") as f:
            data = json.load(f)
            camera_matrix = np.matrix(data["cameraMatrix"])

            if "distCoeffs" in data and data["distCoeffs"] is not None:
                dist_coeffs = np.matrix(data["distCoeffs"])

    if args.input_image is not None:
        image_cap = cv2.imread(args.input_image, cv2.IMREAD_ANYCOLOR)

        tlns = process(image_cap, None, args.descriptor)

        if args.verbose:
            for tln in tlns:
                # if tln.get_max_depth() >= MIN_DEPTH:
                print(tln.get_lhds_name())

    if args.input_image_directory is not None:

        # get all images in dir
        filenames = []
        for f in os.listdir(args.input_image_directory):

            if not f.lower()[-4:] in [".jpg", ".png"]:
                continue

            filenames.append([args.input_image_directory, f])

        # print("loaded {} images in directory: {}".format(len(filenames), args.input_image_directory))

        filenames = sorted(filenames)

        rvecs = []
        tvecs = []

        for filename in filenames:
            image_cap = cv2.imread(os.path.join(*filename), cv2.IMREAD_ANYCOLOR)
            matches = process(image_cap, None, args.descriptor)

            if args.verbose:
                print(filename[1], end=" ")

                for tln in matches:
                    for valid_node_tuple in validate_node(tln):
                        print("{}|{}|{}".format(*valid_node_tuple), end=" ")

                print("")

            if args.descriptor is not None:
                for m in matches:
                    if m.rvec is not None and m.tvec is not None:
                        rvecs.append(m.rvec.tolist())
                        tvecs.append(m.tvec.tolist())
                    else:
                        rvecs.append(None)
                        tvecs.append(None)

                if len(matches) == 0:
                    rvecs.append(None)
                    tvecs.append(None)    


    if args.input_video is not None or args.input_device:

        if args.input_video is not None:
            cap = cv2.VideoCapture(args.input_video)

        if args.input_device is not None:
            cap = cv2.VideoCapture(args.input_device)

        rvecs = []
        tvecs = []

        while(cap.isOpened()):

            _, image_cap = cap.read()

            if image_cap is None:
                break

            matches = process(image_cap, None, args.descriptor)
            for m in matches:
                if m.rvec is not None and m.tvec is not None:
                    rvecs.append(m.rvec.tolist())
                    tvecs.append(m.tvec.tolist())
                else:
                    rvecs.append(None)
                    tvecs.append(None)   

            if len(matches) == 0:
                rvecs.append(None)
                tvecs.append(None)                

        cap.release()

    if args.output_file is not None:
        with open(args.output_file, "w") as f:

            data = {
                "rvec": rvecs,
                "tvec": tvecs
            }

            json.dump(data, f)


