import power
import numpy as np

NEARLY_ZERO = 1E-10

def add_dummy_points(S, R, boundary):

    width = boundary[2]-boundary[0]
    height = boundary[3]-boundary[1]
    new_minmax = [
        boundary[0]-width*2,  # minx
        boundary[1]-height*2, # miny
        boundary[2]+width*2,  # maxx
        boundary[3]+height*2, # maxy
    ]

    boundary_points = np.array([
        [new_minmax[0], new_minmax[1]],
        [new_minmax[0], new_minmax[3]],
        [new_minmax[2], new_minmax[3]],
        [new_minmax[2], new_minmax[1]],
        ])
    S_ag = np.concatenate([S, boundary_points])

    R_ag = np.empty(4)
    R_ag.fill(NEARLY_ZERO)
    R_ag = np.concatenate([R, R_ag])

    return S_ag, R_ag #, boundary_points


def remove_dummy_points(voronoi_cell_map):

    dummy_point_keys = list(voronoi_cell_map.keys())[-4:]

    for key in dummy_point_keys:
        del voronoi_cell_map[key]


def get_power_diagram(S, R, boundary=None): # boundary = [minx, miny, maxx, maxy]

    # caveat: when computing the power diagram on a sparse set of points which are
    # placed along a line the angles of infinite cell edges are nearly identical (= parallel lines).
    # To create polygons cropped on the boundary either additional helper points placed along the line
    # of original points could be introduced after computing the cells, or additional dummy points 
    # with minimum weights can be added to the set of sites for cell computation

    if boundary is not None:
        S, R = add_dummy_points(S, R, boundary)

    tri_list, V = power.get_power_triangulation(S, R)
    voronoi_cell_map = power.get_voronoi_cells(S, V, tri_list)

    if boundary is not None:
        remove_dummy_points(voronoi_cell_map)

    return voronoi_cell_map

if __name__ == "__main__":
    
    boundary = [0, 0, 2000, 1000] # minx, miny, maxx, maxy

    S = np.array([
        [100, 50],
        [200, 50],
        [300, 50.1],
        [400, 50.1],
    ])
    R = np.array([1, 2, 3, 4])

    S, R = add_dummy_points(S, R, boundary)

    tri_list, V = power.get_power_triangulation(S, R)
    voronoi_cell_map = power.get_voronoi_cells(S, V, tri_list)

    remove_dummy_points(voronoi_cell_map)

    power.display(S, R, tri_list, voronoi_cell_map)