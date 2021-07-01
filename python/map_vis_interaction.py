#!/usr/bin/env python

try:
    import lanelet2

    use_lanelet2_lib = True
except ImportError:
    import warnings

    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet


import os
import matplotlib.pyplot as plt
import numpy as np

from utils import map_vis_lanelet2



scenario_name = 'DR_CHN_Merging_ZS'
lat_origin = 0  # origin is necessary to correctly project the lat lon values of the map to the local
lon_origin = 0  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario

def draw_map(senario_name, predicted_data, real_data):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(root_dir, "maps")
    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, scenario_name + lanelet_map_ending)

    # create a figure
    fig, axes = plt.subplots(figsize=(12,6))
    fig.canvas.set_window_title("Interaction Dataset Visualization")

    # load and draw the lanelet2 map, either with or without the lanelet2 library

    print("Loading map...")
    if use_lanelet2_lib:
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
    else:
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)

    for i in predicted_data:
        x = i[:,0]
        y = i[:,1]
        plt.scatter(x, y, marker='.', color='r', label='predicted')
    for i in real_data:
        x = i[:, 0]
        y = i[:, 1]
        plt.scatter(x, y, marker='.', color='b', label='real')

    plt.show()

if __name__ == '__main__':
    xy_1 = np.array(
        [(1067.493, 959.049), (1066.889, 959.071), (1066.286, 959.094), (1065.682, 959.118), (1065.08, 959.142),
         (1064.478, 959.167),
         (1063.877, 959.192), (1063.279, 959.218), (1062.682, 959.245), (1062.087, 959.272), (1061.494, 959.301),
         (1060.903, 959.331)])

    xy_2 = np.array(
        [(1070.356, 945.323), (1072.136, 945.332), (1073.918, 945.348), (1075.702, 945.369), (1077.488, 945.398),
         (1079.275, 945.434),
         (1081.064, 945.478), (1082.854, 945.531), (1084.645, 945.593), (1086.437, 945.666), (1088.231, 945.749),
         (1090.027, 945.843)])

    real_data = np.array([xy_1, xy_2])
    xy_3 = np.array(
        [(1067.493, 959.049), (1066.889, 959.071), (1066.286, 959.094), (1065.682, 959.118), (1065.08, 959.142),
         (1064.478, 959.167),
         (1063.877, 960.192), (1063.279, 960.218), (1062.682, 960.245), (1062.087, 960.272), (1061.494, 960.301),
         (1060.903, 960.331)])

    xy_4 = np.array(
        [(1070.356, 945.323), (1072.136, 945.332), (1073.918, 945.348), (1075.702, 945.369), (1077.488, 945.398),
         (1079.275, 945.434),
         (1081.064, 946.478), (1082.854, 946.531), (1084.645, 946.593), (1086.437, 946.666), (1088.231, 946.749),
         (1090.027, 946.843)])
    predicted_data = np.array([xy_3, xy_4])
    draw_map(scenario_name, predicted_data, real_data)
