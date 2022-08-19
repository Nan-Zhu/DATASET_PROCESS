#!/usr/bin/env python
import math
import random

import lanelet2
import matplotlib
import matplotlib.axes as axes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import glob
import numpy as np
from lanelet2.core import AttributeMap, TrafficLight, Lanelet, LineString3d, Point2d, Point3d, getId, \
    LaneletMap, BoundingBox2d, BasicPoint2d, BasicPoint3d

scenario_name = "DR_USA_Intersection_EP0"
agent_type = {'car': 1, 'truck': 2, 'bus': 3, 'motorcycle': 4,
              'bicycle': 5}
neighbor_distance = 30

# merge lanelet id
merge_to_id = {
    'DR_CHN_Merging_ZS': [30035, 30037],
    'DR_USA_Intersection_EP0': [30030, 30008],
    "DR_CHN_Roundabout_LN": []
}

merge_id = {
    'DR_CHN_Merging_ZS': [30036, 30028],
    'DR_USA_Intersection_EP0': [30022, 30032],
    "DR_CHN_Roundabout_LN": []
}

merge_from_id = {
    'DR_CHN_Merging_ZS': [30031, 30023],
    'DR_USA_Intersection_EP0': [30023, 30044],
    "DR_CHN_Roundabout_LN": []
}

ll_seg_num = 15

########### Class ############
class X:
    def __init__(self, x, y, vx, vy, theta):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.theta = theta

    def __str__(self):
        return 'x：%f, y：%f, vx: %f, vy: %f, theta: %f' %(self.x, self.y, self.vx, self.vy, self.theta)


class F:
    def __init__(self, pts, l, theta):
        self.pts = pts
        self.l = l
        self.theta = theta

class L:
    def __init__(self, Xs, Xe, Xf):
        self.id = None
        self.Xs = Xs
        self.Xe = Xe
        self.Xf = Xf


########### Map Processing ############

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
    assert isinstance(axes, matplotlib.axes.Axes)

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


    #draw_llt(laneletmap.laneletLayer)


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
    object_id_set = list(object_dict.keys())
    if len(object_id_set) > 0:
        ego = BasicPoint2d(object_dict[object_id_set[0]][4], object_dict[object_id_set[0]][5])
    else:
        ego = BasicPoint2d(0, 0)
    for id in object_id_set:

        obj = BasicPoint2d(object_dict[id][4], object_dict[id][5])
        width = object_dict[id][10]
        length = object_dict[id][9]
        psi_rad = object_dict[id][8]
        if object_id_set.index(id) == 0:
            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(obj, width, length, psi_rad), closed=True,
                                          zorder=20, color='r')
        elif lanelet2.geometry.distance(obj, ego) < neighbor_distance:
            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(obj, width, length, psi_rad), closed=True,
                                              zorder=20, color="g")
        else:
            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(obj, width, length, psi_rad), closed=True,
                                              zorder=20)

        axes.add_patch(rect)
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
    plt.plot(pt_x, pt_y, color='#FFFF00', alpha=0.8, linewidth=10)


def draw_llt(llt):  # draw a list of lanelets
    for ll in llt:
        centerline = ll.centerline
        pt_x = [pt.x for pt in centerline]
        pt_y = [pt.y for pt in centerline]
        plt.plot(pt_x, pt_y, marker='x')
        #plt.text( (pt_x[0]+ pt_x[-1]) / 2, (pt_y[0]+pt_y[-1])/2+0.5, str(ll.id), color='b')
        #plt.text(pt_x[0],pt_y[0] , str(ll.id), color='r')
        #plt.scatter(pt_x[0], pt_y[0], color='r')
        #plt.scatter(pt_x[-1], pt_y[-1], color='r')

# geometric calculation
def in_range(x, bound1, bound2):   # x in range of bound1 and bound2 inclusive
    if bound2 < bound1:
        t = bound1
        bound1 = bound2
        bound2 = t
    return bound1 <= x <= bound2

def getY(x, p1, p2):   # get y of x on line through p1, p2
    a = p2.y - p1.y
    b = p1.x - p2.x
    c = p1.y * p2.x - p1.x * p2.y
    if b == 0:
        return p1.y
    return (-c - a * x) / b

def getX(y, p1, p2):
    a = p2.y - p1.y
    b = p1.x - p2.x
    c = p1.y * p2.x - p1.x * p2.y
    if a == 0:
        return p1.x
    return (-c - b * y) / a

def get_p_on_line(p0, line):    # get point with same x of p0 on line
    if abs(line[-1].x - line[0].x) > abs(line[-1].y - line[0].y):
        for i in range(len(line)-1):
            if in_range(p0.x, line[i].x, line[i+1].x):
                return BasicPoint2d(p0.x, getY(p0.x, line[i], line[i+1]))
    else:
        for i in range(len(line)-1):
            if in_range(p0.y, line[i].y, line[i+1].y):
                return BasicPoint2d(getX(p0.y, line[i], line[i+1]), p0.y)
    p_f = BasicPoint2d(line[0].x, line[0].y)
    p_e = BasicPoint2d(line[-1].x, line[-1].y)
    return p_f if lanelet2.geometry.distance(p0, p_f) < lanelet2.geometry.distance(p0, p_e) else p_e


def get_theta(p1, p2):  # get angle of line p1,p2 with x
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    theta = math.atan2(dy, dx)
    return theta


def get_ll_theta(ll):  # get angle of lanelet
    p1 = ll.centerline[0]
    p2 = ll.centerline[-1]
    return get_theta(p1, p2)


def get_nearest_centerline(map, pt, theta=None):  # get nearest lanelet based on centerline
    pt = Point3d(getId(), pt.x, pt.y, 0)
    min_dis = 10e9
    nearest_ll = None
    for ll in map.laneletLayer:
        cl = ll.centerline
        if lanelet2.geometry.distance(pt, cl) < min_dis:
            if theta:
                if abs(get_ll_theta(ll) - theta) < math.pi / 4:
                    min_dis = lanelet2.geometry.distance(pt, cl)
                    nearest_ll = ll
            else:
                min_dis = lanelet2.geometry.distance(pt, cl)
                nearest_ll = ll
    return nearest_ll


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
    if scenario_name == 'DR_CHN_Merging_ZS':
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
    e_id = 0
    Xe = None
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
                        e_id = id2
                        Xe = X(t.x, t.y, object_dict[id2][6], object_dict[id2][7], object_dict[id2][8])
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
            if llt1 == llt and len(dia) == 1:
                if llt1.id in merge_id[scenario_name]:   # llt merges at merge pt
                    if scenario_name == 'DR_USA_Intersection_EP0':
                        dia[0] = merge_pt[llt1.id]
                    else:
                        dia.append(merge_pt[llt1.id])
                        Xe = X(merge_pt[llt1.id].x, merge_pt[llt1.id].y, 0, 0, 0)
                else:
                    for pt in centerline:
                        if tail:
                            # p1 = BasicPoint2d(pt.x, pt.y)
                            # h = BasicPoint2d(start.x, start.y)
                            # t = BasicPoint2d(tail.x, tail.y)
                            #if lanelet2.geometry.distance(h, p1) < lanelet2.geometry.distance(t, p1):
                            if abs(centerline[0].x-centerline[-1].x) > abs(centerline[0].y-centerline[-1].y):
                                if pt.x < start.x < tail.x or pt.x > start.x > tail.x:
                                    dia.append(pt)
                                    Xe = X(pt.x, pt.y, 0, 0, 0)
                            elif pt.y < start.y < tail.y or pt.y > start.y > tail.y:
                                dia.append(pt)
                                Xe = X(pt.x, pt.y, 0, 0, 0)
                        else:
                            dia.append(pt)

            # loop next llt
            tollt = llt1
            llts_following = graph.following(llt1, False)
            if len(llts_following) == 1:
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
                            if ll.id in merge_id[scenario_name]:  # llt merges at merge pt
                                if scenario_name == 'DR_USA_Intersection_EP0':
                                    dia[0] = merge_pt[llt1.id]
                                else:
                                    dia.append(merge_pt[llt1.id])
                                    Xe = X(merge_pt[llt1.id].x, merge_pt[llt1.id].y, 0, 0, 0)
                            else:
                                for pt in ll.centerline:
                                    dia.append(pt)
                                    Xe = X(pt.x, pt.y, 0, 0, 0)
    return dia, Xe, e_id


# formatting dia with evenly spaced n segments
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
    if len(new_dia) == n:
        new_dia.append(dia[-1])
    theta = 0
    if len(new_dia) > n:
        theta = get_theta(new_dia[int(n/2)], new_dia[int(n/2)+1])
    draw_dia(new_dia)
    Xf = F(new_dia, length, theta)
    return Xf


def get_all_dia(map, object_dict):
    # set lanelet merge point
    merge_pt = dict()
    if scenario_name == 'DR_CHN_Merging_ZS':
        merge_pt[merge_id[scenario_name][0]] = map.laneletLayer[30031].centerline[3]
        merge_pt[merge_id[scenario_name][1]] = map.laneletLayer[30023].centerline[5]
    if scenario_name == 'DR_USA_Intersection_EP0':
        merge_pt[merge_id[scenario_name][0]] = map.laneletLayer[30030].centerline[0]
        merge_pt[merge_id[scenario_name][1]] = map.laneletLayer[30014].centerline[0]

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
        if abs(get_ll_theta(nearestllt_h) - get_theta(tail, head)) > math.pi / 4:
            nearestllt_h = get_nearest_centerline(map, head, get_theta(tail, head))
            if not nearestllt_h:
                nearestllt_h = get_nearest_centerline(map, head)

        if abs(get_ll_theta(nearestllt_t) - get_theta(tail, head)) > math.pi / 4:
            nearestllt_t = get_nearest_centerline(map, tail, get_theta(tail, head))
            if not nearestllt_t:
                nearestllt_t = get_nearest_centerline(map, tail)

        head_on_llt[nearestllt_h.id].append(id)
        tail_on_llt[nearestllt_t.id].append(id)

    # get all dia from map
    dia_list = list()
    L_list = list()
    # start from object head
    for id in object_dict.keys():
        head, tail = get_obj_shape(id, object_dict)
        Xs = X(head.x, head.y, object_dict[id][6], object_dict[id][7], object_dict[id][8])
        llt = lanelet2.geometry.findNearest(map.laneletLayer, head, 1)[0][1]
        if abs(get_ll_theta(llt) - get_theta(tail, head)) > math.pi / 4:
            llt = get_nearest_centerline(map, head, get_theta(tail, head))
            if not llt:
                llt = get_nearest_centerline(map, head)
        dia, Xe, e_id = get_dia(graph, head, tail, llt, tail_on_llt, object_dict, merge_pt)
        Xf = get_control_point(ll_seg_num, dia)
        Li = L(Xs, Xe, Xf)
        Li.id = id
        L_list.append(Li)
        dia_list.append(dia)

    tmp_id = 1
    # start from lanelet starting point
    for ll in map.laneletLayer:
        llts_previous = graph.previous(ll, False)
        if len(llts_previous) == 0:
            centerline = ll.centerline
            start = BasicPoint2d(centerline[0].x, centerline[0].y)
            if scenario_name == 'DR_USA_Intersection_EP0' and ll.id in merge_id[scenario_name]:
                Xs = X(merge_pt[ll.id].x, merge_pt[ll.id].y, 0, 0, 0)
            else:
                Xs = X(start.x, start.y, 0, 0, 0)
            dia, Xe, e_id = get_dia(graph, start, None, ll, tail_on_llt, object_dict, merge_pt)
            Xf = get_control_point(ll_seg_num, dia)
            Li = L(Xs, Xe, Xf)
            Li.id = tmp_id * 10000 + e_id
            tmp_id = tmp_id + 1
            L_list.append(Li)
            dia_list.append(dia)

    # start from merge point
    if scenario_name == 'DR_CHN_Merging_ZS':
        for i in range(len(merge_id)):
            ll = map.laneletLayer[merge_from_id[i]]
            start = BasicPoint2d(merge_pt[merge_id[i]].x, merge_pt[merge_id[i]].y)
            Xs = X(start.x, start.y, 0, 0, 0)
            dia, Xe, e_id = get_dia(graph, start, None, ll, tail_on_llt, object_dict, merge_pt)
            Xf = get_control_point(ll_seg_num, dia)
            Li = L(Xs, Xe, Xf)
            Li.id = tmp_id * 10000 + e_id
            tmp_id = tmp_id + 1
            L_list.append(Li)
            dia_list.append(dia)

    #start from fork point
    for llt in laneletmap.laneletLayer:
        llts_following = graph.following(llt, False)
        if len(llts_following) > 1:
            for ll in llts_following:
                centerline = ll.centerline
                start = BasicPoint2d(centerline[0].x, centerline[0].y)
                #plt.scatter(start.x, start.y, color='r')
                Xs = X(start.x, start.y, 0, 0, 0)
                dia, Xe, e_id = get_dia(graph, start, None, ll, tail_on_llt, object_dict, merge_pt)
                Xf = get_control_point(ll_seg_num, dia)
                # Xf = None
                # draw_dia(dia)
                Li = L(Xs, Xe, Xf)
                Li.id = tmp_id * 10000 + e_id
                tmp_id = tmp_id + 1
                L_list.append(Li)
                dia_list.append(dia)

    obj_dia_dict = get_dia_of_objects(dia_list, map, graph, object_dict, L_list)
    return obj_dia_dict


#  get dia of objects within neighbour distance and can be reached
def get_dia_of_objects(dia_list, map, graph, object_dict, L_list):
    obj_dia_dict = dict()
    for id in object_dict.keys():
        c = np.random.rand(3, )
        obj_dia_dict[id] = list()
        head, tail = get_obj_shape(id, object_dict)
        llt1 = lanelet2.geometry.findNearest(map.laneletLayer, head, 1)[0][1]
        if abs(get_ll_theta(llt1) - get_theta(tail, head)) > math.pi / 4:
            llt1 = get_nearest_centerline(map, head, get_theta(tail, head))
            if not llt1:
                llt1 = get_nearest_centerline(map, head)

        for i in range(len(dia_list)):
            choose = False
            dia = dia_list[i]
            start_pt = BasicPoint2d(dia[0].x, dia[0].y)
            if lanelet2.geometry.distance(head,start_pt) > 20:
                continue
            for pt in dia:
                pt = BasicPoint2d(pt.x, pt.y)
                llt2 = lanelet2.geometry.findNearest(map.laneletLayer, pt, 1)[0][1]
                # dia in front of object
                if lanelet2.geometry.distance(pt, head) <= lanelet2.geometry.distance(pt, tail):
                    if hasPathFromTo(graph, llt1, llt2):
                        if scenario_name == 'DR_USA_Intersection_EP0':
                            if lanelet2.geometry.distance(pt, head) < neighbor_distance and abs(L_list[i].Xf.theta - get_theta(tail, head)) < math.pi / 2:
                                choose = True
                                break
                        else:
                            choose = True
                            break
                # dia behind object
                # else:
                #     if hasPathFromTo(graph, llt2, llt1):
                #         route = graph.getRoute(llt2, llt1)
                #         if route:
                #             path = route.shortestPath()
                #             for ll in path:
                #                 if scenario_name == 'DR_CHN_Merging_ZS' and ll.id in merge_from_id:  # path through merge lanelet
                #                     choose = False
                #                     break
                #         if not is_ego_lane(map, graph, llt1, llt2):  # not ego lane
                #             choose = True

            if choose:
                min_dis = 10e9
                for pt in dia:
                    pt = BasicPoint2d(pt.x, pt.y)
                    if lanelet2.geometry.distance(head, pt) < min_dis:
                        min_dis = lanelet2.geometry.distance(head, pt)
                obj_dia_dict[id].append([L_list[i], min_dis])
                for pt in dia:
                    pt = BasicPoint2d(pt.x, pt.y)
                    if lanelet2.geometry.distance(head, pt) <= neighbor_distance:
                        obj_dia_dict[id].append(dia)
                        if list(object_dict.keys()).index(id) == 0:
                            draw_dia_of_object(dia, c)
                        break
        head, tail = get_obj_shape(id, object_dict)
        plt.scatter(head.x, head.y, color='b')

    return obj_dia_dict


def to_array(f, id, dia):
    dis = dia[1]
    dia = dia[0]
    Xs = dia.Xs
    Xe = dia.Xe
    Xf = dia.Xf

    #print(dia.id)
    #print(Xs)
    #print(Xe)
    arr = [f, id, dia.id, Xs.x, Xs.y, Xs.vx, Xs.vy, Xs.theta, Xe.x, Xe.y, Xe.vx, Xe.vy, Xe.theta]
    for pt in Xf.pts:
        arr.append(pt.x)
        arr.append(pt.y)
    arr.append(Xf.l)
    arr.append(Xf.theta)
    arr.append(dis)
    return arr

if __name__ == '__main__':
    # create dir
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(root_dir, "maps")
    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, scenario_name + lanelet_map_ending)

    # read object data
    file_path_list = sorted(glob.glob(os.path.join(root_dir, 'recorded_trackfiles/' + scenario_name + '/vehicle*.csv')))

    print('Generating DIA Data...')

    # for file_path in file_path_list:
    #     now_dict = get_frame_instance_dict(file_path)
    #     frame_id_set = sorted(set(now_dict.keys()))
    #     f = open(file_path[:-4]+'_dia.csv', 'w')
    #     projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
    #     laneletmap = lanelet2.io.load(lanelet_map_file, projector)
    #     for frame in frame_id_set:
    #         print("frame" + str(int(frame)))
    #         obj_dia_dict = get_all_dia(laneletmap, now_dict[int(frame)])
    #         for id in obj_dia_dict.keys():
    #             for dia in obj_dia_dict[id]:
    #                 arr = to_array(frame, id, dia)
    #                 f.write(','.join(str(x) for x in arr)+'\n')
    #     f.close()


    for i in range(50):
        fig, axes = plt.subplots(1, 1)
        fig.canvas.set_window_title("Interaction Dataset Visualization")
        print("Loading map...", i)
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        now_dict = get_frame_instance_dict(file_path_list[0])
        frame_id_set = sorted(set(now_dict.keys()))
        draw_lanelet_map(laneletmap, axes)
        f = random.randint(1, 1000)
        draw_object(now_dict[frame_id_set[f]])
        obj_dia_dict = get_all_dia(laneletmap, now_dict[frame_id_set[f]])
        fig.set_size_inches(22, 16)
        plt.savefig(root_dir+"/plots/frame"+str(f)+".png")
        #plt.show()


