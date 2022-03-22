#!/usr/bin/env python
import random

import lanelet2
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import glob
import numpy as np
from lanelet2.core import AttributeMap, TrafficLight, Lanelet, LineString3d, Point2d, Point3d, getId, \
    LaneletMap, BoundingBox2d, BasicPoint2d

scenario_name = "DR_CHN_Merging_ZS"
agent_type = {'car': 1, 'truck': 2, 'bus': 3, 'motorcycle': 4,
              'bicycle': 5}
neighbor_distance = 30

# merge lanelet id
merge_to_id = [30035, 30037]
merge_id = [30036, 30028]
merge_from_id = [30031, 30023]


########### Map Visualization ############
def set_visible_area(laneletmap, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for point in laneletmap.pointLayer:
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])


def draw_lanelet_map(laneletmap, axes):

    set_visible_area(laneletmap, axes)

    unknown_linestring_types = list()

    # set color for map elements
    for ls in laneletmap.lineStringLayer:
        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=2, zorder=10)
        elif ls.attributes["type"] == "pedestrian_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "bike_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "stop_line":
            type_dict = dict(color="white", linewidth=3, zorder=10)
        elif ls.attributes["type"] == "virtual":
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
        elif ls.attributes["type"] == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=2)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=2)

        else:
            if ls.attributes["type"] not in unknown_linestring_types:
                unknown_linestring_types.append(ls.attributes["type"])
            continue
        ls_points_x = [pt.x for pt in ls]
        ls_points_y = [pt.y for pt in ls]

        plt.plot(ls_points_x, ls_points_y, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

    axes.patch.set_facecolor("lightgray")

    # draw keep out area
    areas = []
    for area in laneletmap.areaLayer:
        if area.attributes["subtype"] == "keepout":
            points = [[pt.x, pt.y] for pt in area.outerBoundPolygon()]
            polygon = Polygon(points, True)
            areas.append(polygon)

    area_patches = PatchCollection(areas, facecolors="darkgray", edgecolors="None", zorder=5)
    axes.add_collection(area_patches)


########## Read file ###############
def get_frame_instance_dict(pra_file_path):
    '''
	Read raw data from files and return a dictionary:
		{frame_id:
			{object_id:
				# 11 features
				[track_id, frame_id, timestamp_ms, agent_type, x, y, vx, vy, psi_rad, length, width]
			}
		}
		object_type: agent_type{ car: 1,  trunk: 2}
	'''
    with open(pra_file_path, 'r') as reader:
        # print(train_file_path)
        reader.readline()
        content = np.array([x.strip().split(',') for x in reader.readlines()]).astype(str)
        now_dict = {}
        for row in content:
            row[3] = agent_type[row[3]]
        content = content.astype(float)
        for row in content:
            n_dict = now_dict.get(row[1], {})
            n_dict[row[0]] = row  # [2:]
            now_dict[row[1]] = n_dict
    return now_dict


########### Object Processing ############
def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length, psi_rad):
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=psi_rad)


def draw_object(object_dict):
    object_id_set = object_dict.keys()
    for id in object_id_set:
        obj = Point2d(getId(), object_dict[id][4], object_dict[id][5])
        width = object_dict[id][10]
        length = object_dict[id][9]
        psi_rad = object_dict[id][8]
        rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(obj, width, length, psi_rad), closed=True,
                                          zorder=20)
        axes.add_patch(rect)
        if id % 20 == 0:
            axes.text(obj.x, obj.y + 1, str(int(id)),color='r', horizontalalignment='center', zorder=30)
        else:
            axes.text(obj.x, obj.y + 1, str(int(id)), horizontalalignment='center', zorder=30)


def get_obj_shape(id, object_dict):   # get head and tail of object
    obj = BasicPoint2d( object_dict[id][4], object_dict[id][5])
    width = object_dict[id][10]
    length = object_dict[id][9]
    psi_rad = object_dict[id][8]
    shape = polygon_xy_from_motionstate(obj, width, length, psi_rad)
    head = BasicPoint2d((shape[1][0] + shape[2][0]) / 2, (shape[1][1] + shape[2][1]) / 2)
    tail = BasicPoint2d((shape[0][0] + shape[3][0]) / 2, (shape[0][1] + shape[3][1]) / 2)
    return head, tail


########### Dia Processing ############
# plot function
def draw_dia(line):  # draw all dia
    pt_x = [pt.x for pt in line if pt is not None]
    pt_y = [pt.y for pt in line if pt is not None]
    plt.plot(pt_x, pt_y, color='k', alpha=0.2, linewidth=10)


def draw_dia_of_object(line, c):  # draw dia relating to an object
    pt_x = [pt.x for pt in line if pt is not None]
    pt_y = [pt.y for pt in line if pt is not None]
    plt.plot(pt_x, pt_y, color=c, alpha=0.5, linewidth=10)


def draw_llt(llt):  # draw a list of lanelets
    for ll in llt:
        centerline = ll.centerline
        pt_x = [pt.x for pt in centerline]
        pt_y = [pt.y for pt in centerline]
        plt.plot(pt_x, pt_y, marker = 'x')


# geometric calculation
def in_range(x, bound1, bound2):   # x in range of bound1 and bound2 inclusive
    if bound2 < bound1:
        t = bound1
        bound1 = bound2
        bound2 = t
    return bound1 <= x <= bound2


def getY(x, p1, p2):  # get y of x on line through p1, p2
    a = p2.y - p1.y
    b = p1.x - p2.x
    c = p1.y * p2.x - p1.x * p2.y
    if b == 0:
        return p1.y
    return (-c - a * x) / b


def get_p_on_line(p0, line):  # get point with same x of p0 on line
    for i in range(len(line)-1):
        if in_range(p0.x, line[i].x, line[i+1].x):
            return BasicPoint2d(p0.x, getY(p0.x, line[i], line[i+1]))


# lanelet relation judgement
def hasPathFromTo(graph, start, target):  # judge if there is a path from start llt to target llt
    class TargetFound(BaseException):
        pass

    def raiseIfDestination(visitInformation):
        if visitInformation.lanelet == target:
            raise TargetFound()
        else:
            return True
    try:
        graph.forEachSuccessor(start, raiseIfDestination)
        return False
    except TargetFound:
        return True


def is_previous(graph, llt1, llt2):  # judge if llt2 is previous lanelet of llt1
    if llt1 == llt2:
        return True
    prev = graph.previous(llt1, False)
    for llt_prev in prev:
        if llt_prev == llt2:
            return True
        return is_previous(graph, llt_prev, llt2)
    return False


def is_following(graph, llt1, llt2):   # judge if llt2 is following lanelet of llt1
    if llt1 == llt2:
        return True
    follow = graph.following(llt1, False)
    for llt_f in follow:
        if llt_f == llt2:
            return True
        return is_following(graph, llt_f, llt2)
    return False


def is_ego_lane(map, graph, llt1, llt2):   # judge if llt2 is among ego lane of llt1
    if llt1 == llt2:
        return True
    for i in range(len(merge_id)):
        if is_following(graph, map.laneletLayer[merge_to_id[i]], llt1) and is_previous(graph, map.laneletLayer[merge_id[i]], llt2):
            return True
    prev = graph.previous(llt1, False)
    for llt_prev in prev:
        if llt_prev == llt2:
            return True
        return is_ego_lane(map, graph, llt_prev, llt2)
    return False


# get dia from start
def get_dia(graph, start, tail, llt, tail_on_llt, object_dict, merge_pt):
    dia = list()
    stop = False
    min_dis = 10e9
    next_tail = None
    centerline = llt.centerline

    if tail:    # if start from object, find point with same x on line
        p = get_p_on_line(start, centerline)
        dia.append(p) if p else dia.append(start)
    else:
        dia.append(start)

    llt1 = llt
    centerline1 = centerline

    fromllt = None  # following lanelet of start
    if not len(graph.following(llt, False)) == 0:
        fromllt = graph.following(llt, False)[0]
    tollt = None  # previous lanelet of end

    # loop until front object tail or lanelet end
    while not stop:
        # find front object
        if len(tail_on_llt[llt1.id]) > 0:
            for id2 in tail_on_llt[llt1.id]:
                h, t = get_obj_shape(id2, object_dict)
                if not start == h:
                    if lanelet2.geometry.distance(start, t) < lanelet2.geometry.distance(start, h) < min_dis:
                        min_dis = lanelet2.geometry.distance(start, t)
                        next_tail = t

            if next_tail:
                stop = True
                # add lanelets between
                if fromllt and tollt:
                    route = graph.getRoute(fromllt, tollt)
                    if (route):
                        path = route.shortestPath()
                        for ll in path:
                            for pt in ll.centerline:
                                dia.append(pt)
                # add pt before next tail
                for pt in centerline1:
                    if in_range(pt.x, start.x, next_tail.x) and in_range(pt.y, start.y, next_tail.y):
                        dia.append(pt)
                # add pt with same x of next tail on line
                p = get_p_on_line(next_tail, centerline1)
                dia.append(p) if p else dia.append(next_tail)

        # no front object on the lanelet
        if not stop:
            # add pt on lanelet after start
            if llt1 == llt:
                if llt1.id in merge_id:   # llt merges at merge pt
                    dia.append(merge_pt[llt1.id])
                else:
                    for pt in centerline:
                        if tail:
                            if tail.x < start.x < pt.x or tail.x > start.x > pt.x:
                                dia.append(pt)

            # loop next llt
            tollt = llt1
            llts_following = graph.following(llt1, False)
            if not len(llts_following) == 0:
                llt1 = llts_following[0]
                centerline1 = llt1.centerline
            else:   # lanelet end
                stop = True
                # get shortest path between fromllt and tollt
                if fromllt and tollt:
                    route = graph.getRoute(fromllt, tollt)
                    if route:
                        path = route.shortestPath()
                        for ll in path:
                            if ll.id in merge_id:  # llt merges at merge pt
                                dia.append(merge_pt[llt1.id])
                            else:
                                for pt in ll.centerline:
                                    dia.append(pt)
    return dia


# formatting dia with evenly spaced n control points
def get_control_point(n, dia):
    new_dia = list()
    new_dia.append(dia[0])
    length = 0
    for i in range(len(dia)-1):
        p1 = BasicPoint2d(dia[i].x, dia[i].y)
        p2 = BasicPoint2d(dia[i+1].x, dia[i+1].y)
        length = length + lanelet2.geometry.distance(p1, p2)
    seg = length / n
    l = seg
    for i in range(len(dia) - 1):
        lx = dia[i].x
        p1 = BasicPoint2d(dia[i].x, dia[i].y)
        p2 = BasicPoint2d(dia[i + 1].x, dia[i + 1].y)
        tl = lanelet2.geometry.distance(p1, p2)
        while tl > 0:
            if tl < l:
                l = l - tl
                tl = 0
            else:
                tx = lx + (l / tl) * (p2.x - lx)
                p = BasicPoint2d(tx, getY(tx, p1, p2))
                new_dia.append(p)
                tl = tl - l
                l = seg
                lx = tx
    new_dia.append(dia[-1])
    draw_dia(new_dia)
    return new_dia


def get_all_dia(map, object_dict):
    # set lanelet merge point
    merge_pt = dict()
    merge_pt[merge_id[0]] = map.laneletLayer[30031].centerline[3]
    merge_pt[merge_id[1]] = map.laneletLayer[30023].centerline[5]

    # build graph from map
    traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                  lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(map, traffic_rules)

    # get dict of objects head and tail on certain lanelet
    head_on_llt = dict()
    tail_on_llt = dict()
    for ll in laneletmap.laneletLayer:
        head_on_llt[ll.id] = list()
        tail_on_llt[ll.id] = list()
    for id in object_dict.keys():
        head, tail = get_obj_shape(id, object_dict)
        nearest_h = lanelet2.geometry.findNearest(map.laneletLayer, head, 1)
        nearest_t = lanelet2.geometry.findNearest(map.laneletLayer, tail, 1)
        nearestllt_h = nearest_h[0][1]
        nearestllt_t = nearest_t[0][1]
        head_on_llt[nearestllt_h.id].append(id)
        tail_on_llt[nearestllt_t.id].append(id)

    # get all dia from map
    dia_list = list()

    # start from object head
    for id in object_dict.keys():
        head, tail = get_obj_shape(id, object_dict)
        llt = lanelet2.geometry.findNearest(map.laneletLayer, head, 1)[0][1]
        dia = get_dia(graph, head, tail, llt, tail_on_llt, object_dict, merge_pt)
        dia_list.append(dia)

    # start from lanelet starting point
    for ll in map.laneletLayer:
        llts_previous = graph.previous(ll, False)
        if len(llts_previous) == 0:
            centerline = ll.centerline
            start = BasicPoint2d(centerline[0].x, centerline[0].y)
            dia = get_dia(graph, start, None, ll, tail_on_llt, object_dict, merge_pt)
            dia_list.append(dia)

    # start from merge point
    for i in range(len(merge_id)):
        ll = map.laneletLayer[merge_from_id[i]]
        start = BasicPoint2d(merge_pt[merge_id[i]].x, merge_pt[merge_id[i]].y)
        dia = get_dia(graph, start, None, ll, tail_on_llt, object_dict, merge_pt)
        dia_list.append(dia)

    new_dia_list = list()
    for dia in dia_list:
        new_dia_list.append(get_control_point(5, dia))

    obj_dia_dict = get_dia_of_objects(new_dia_list, map, graph, object_dict)
    return obj_dia_dict


#  get dia of objects within neighbour distance and can be reached
def get_dia_of_objects(new_dia_list, map, graph, object_dict):
    obj_dia_dict = dict()
    for id in object_dict.keys():
        c = np.random.rand(3, )
        obj_dia_dict[id] = list()
        head, tail = get_obj_shape(id, object_dict)
        llt1 = lanelet2.geometry.findNearest(map.laneletLayer, head, 1)[0][1]

        for dia in new_dia_list:
            choose = False
            for pt in dia:
                pt = BasicPoint2d(pt.x, pt.y)
                llt2 = lanelet2.geometry.findNearest(map.laneletLayer, pt, 1)[0][1]
                # dia in front of object
                if lanelet2.geometry.distance(pt, head) <= lanelet2.geometry.distance(pt, tail):
                    if hasPathFromTo(graph, llt1, llt2):
                        choose = True
                        break
                # dia behind object
                else:
                    if hasPathFromTo(graph, llt2, llt1):
                        route = graph.getRoute(llt2, llt1)
                        if route:
                            path = route.shortestPath()
                            for ll in path:
                                if ll.id in merge_from_id: # path through merge lanelet
                                    choose = False
                                    break
                        if not is_ego_lane(map, graph, llt1, llt2): # not ego lane
                            choose = True

            if choose:
                for pt in dia:
                    pt = BasicPoint2d(pt.x, pt.y)
                    if lanelet2.geometry.distance(head, pt) <= neighbor_distance:
                        obj_dia_dict[id].append(dia)
                        if id % 20 == 0:
                            draw_dia_of_object(dia, c)
                        break

    return obj_dia_dict


if __name__ == '__main__':
    # create dir
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(root_dir, "maps")
    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, scenario_name + lanelet_map_ending)

    # read object data
    file_path_list = sorted(glob.glob(os.path.join(root_dir, 'dataset/INTERACTION/recorded_trackfiles/' + scenario_name + '/*000.csv')))

    now_dict = {}
    frame_id_set = {}
    for file_path in file_path_list:
        now_dict = get_frame_instance_dict(file_path)
        frame_id_set = sorted(set(now_dict.keys()))

    print('Generating DIA Data...')

    # save plots
    for i in range(10):
        fig, axes = plt.subplots(1, 1)
        fig.canvas.set_window_title("Interaction Dataset Visualization")
        print("Loading map...", i)
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        draw_lanelet_map(laneletmap, axes)
        f = random.randint(1, 100)
        draw_object(now_dict[frame_id_set[f]])
        obj_dia_dict = get_all_dia(laneletmap, now_dict[frame_id_set[f]])
        fig.set_size_inches(22, 16)
        plt.savefig(root_dir+"/plots/frame"+str(f)+".png")
        #plt.show()