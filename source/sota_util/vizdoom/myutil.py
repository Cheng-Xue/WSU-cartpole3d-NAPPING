"""
Vincent Lombardi
"""
from torch import nn
import torch
import numpy as np
import math


# 90 degree angle north


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)




def in_center(e, p):
    if 60 > e['y_position'] > -60 and 55 > p['y_position'] > -55:
        return True
    elif 60 > e['x_position'] > -60 and 55 > p['x_position'] > -55:
        return True
    return False


def in_center2(p):
    if 50 > p['y_position'] > -50 or 50 > p['x_position'] > -50:
        return True
    return False

def in_center3(p):
    if 280 > p['y_position'] > -280 and 280 > p['x_position'] > -280:
        return True
    return False
    
def h_check(e):
    e_health = int(e['health'])
    if e_health == 10:
        # print("1")
        return 3
    elif e_health == 6:
        # print("2")
        return 2
    elif e_health == 2:
        # print("3")
        return 1
    else:
        # print("4")
        return 0

def ob_help(ob_list, player, e_dist, offset):
    for o in ob_list:
        w = 40.0
        if tracker(o) != 1:
            w = 25.0
        if gunner(o, player, offset, w):
            if e_dist > get_dist(player, o):
                return False
    return True
 
 


def to_border(player, target):
    my_act = np.dtype('int64').type(3)
    angle = player['angle']
    if target > 1:
        if target == 2:
            if angle != 90.0:
                if 270 > angle > 90:
                    my_act = np.dtype('int64').type(5)  # 'turn_right'
                else:
                    my_act = np.dtype('int64').type(4)

        elif target == 3:
            if angle != 270.0:
                if 270 > angle > 90:
                    my_act = np.dtype('int64').type(4)  # 'turn_left'
                else:
                    my_act = np.dtype('int64').type(5)
        elif target == 4:
            if angle != 0.0:
                if angle > 180:
                    my_act = np.dtype('int64').type(4)  # 'turn_left'
                else:
                    my_act = np.dtype('int64').type(5)

        elif target == 5:
            if angle != 180.0:
                if angle > 180:
                    my_act = np.dtype('int64').type(5)  # 'turn_right'
                else:
                    my_act = np.dtype('int64').type(4)

        if my_act == 3 and not (
                in_center2(player)):  # or (tracker2(player) == 6 and get_dist(player, self.c_list[target-1]) > 200)):
            if target == 2:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(1)
                else:
                    my_act = np.dtype('int64').type(0)

            elif target == 3:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(0)
                else:
                    my_act = np.dtype('int64').type(1)

            elif target == 4:

                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(0)
                else:
                    my_act = np.dtype('int64').type(1)

            elif target == 5:
                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(1)
                else:
                    my_act = np.dtype('int64').type(0)

    return my_act


def to_center(player, p_coord):
    my_act = np.dtype('int64').type(3)  # 'nothing'  # 'forward'
    angle = player['angle']
    if p_coord == 2:
        if angle != 270.0:
            if 270 > angle > 90:
                my_act = np.dtype('int64').type(4)  # 'turn_left'
            else:
                my_act = np.dtype('int64').type(5)

    elif p_coord == 3:
        if angle != 90.0:
            if 270 > angle > 90:
                my_act = np.dtype('int64').type(5)  # 'turn_right'
            else:
                my_act = np.dtype('int64').type(4)
    elif p_coord == 4:
        if angle != 180.0:
            if angle > 180:
                my_act = np.dtype('int64').type(5)  # 'turn_right'
            else:
                my_act = np.dtype('int64').type(4)

    elif p_coord == 5:
        if angle != 0.0:
            if angle > 180:
                my_act = np.dtype('int64').type(4)  # 'turn_left'
            else:
                my_act = np.dtype('int64').type(5)

    if my_act == 3:  # 'nothing':
        if not in_center2(player):
            if p_coord == 2:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(0)  # 'left'
                else:
                    my_act = np.dtype('int64').type(1)  # 'right'

            elif p_coord == 3:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(1)  # 'right'
                else:
                    my_act = np.dtype('int64').type(0)  # 'left'

            elif p_coord == 4:

                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(1)  # 'right'
                else:
                    my_act = np.dtype('int64').type(0)  # 'left'

            elif p_coord == 5:
                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(0)  # 'left'
                else:
                    my_act = np.dtype('int64').type(1)  # 'right'

    return my_act


def corner(player, p_coord):
    my_act = np.dtype('int64').type(3)  # 'nothing'  # 'forward'

    if p_coord == 2:
        if player['angle'] != 270.0:
            my_act = np.dtype('int64').type(5)  # 'turn_left'

    elif p_coord == 3:
        if player['angle'] != 90.0:
            my_act = np.dtype('int64').type(5)  # 'turn_left'

    elif p_coord == 4:
        if player['angle'] != 180.0:
            my_act = np.dtype('int64').type(5)  # 'turn_left'

    elif p_coord == 5:
        if player['angle'] != 0.0:
            my_act = np.dtype('int64').type(5)  # 'turn_left'

    if my_act == 3:  # 'nothing':
        if not in_center2(player):
            if p_coord == 2:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(0)  # 'left'
                else:
                    my_act = np.dtype('int64').type(1)  # 'right'

            elif p_coord == 3:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(1)  # 'right'
                else:
                    my_act = np.dtype('int64').type(0)  # 'left'

            elif p_coord == 4:

                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(1)  # 'right'
                else:
                    my_act = np.dtype('int64').type(0)  # 'left'

            elif p_coord == 5:
                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(0)  # 'left'
                else:
                    my_act = np.dtype('int64').type(1)  # 'right'

    return my_act
 

    
    
def gunner(e, p, offset, w=18):  # 18 guaranteed
    p_x = p['x_position']
    p_y = p['y_position']
    e_x = e['x_position']
    e_y = e['y_position']

    angle = p['angle'] + offset
    if angle < 0:
        angle = 360 + angle
    elif angle > 360:
        angle = angle - 360
    elif angle == 360:
        angle = 0
    if angle == 90 and e_y > p_y:  # north
        if (p_x - w) < e_x < (p_x + w):
            return True

    elif angle == 0 and e_x > p_x:  # east
        if (p_y - w) < e_y < (p_y + w):
            return True
    elif angle == 270 and e_y < p_y:  # south
        if (p_x - w) < e_x < (p_x + w):
            return True
    elif angle == 180 and e_x < p_x:  # west
        if (p_y - w) < e_y < (p_y + w):
            return True

    elif angle == 45 and e_x > p_x and e_y > p_y:
        dif = e_x - p_x
        if (p_y + dif - w) < e_y < (p_y + dif + w):
            return True
    elif angle == 225 and e_x < p_x and e_y < p_y:
        dif = p_x - e_x
        if (p_y - dif - w) < e_y < (p_y - dif + w):
            return True

    elif angle == 135 and e_x < p_x and e_y > p_y:
        dif = p_x - e_x

        if (p_y + dif - w) < e_y < (p_y + dif + w):
            return True

    elif angle == 315 and e_x > p_x and e_y < p_y:
        dif = e_x - p_x

        if (p_y - dif - w) < e_y < (p_y - dif + w):
            return True

    return False


def target_sighted(e, p):
    p_x = p['x_position']
    p_y = p['y_position']
    e_x = e['x_position']
    e_y = e['y_position']
    if e_y > 352 and p_y > 352:
        return True
    elif e_y < -352 and p_y < -352:
        return True
    elif e_x < -352 and p_x < -352:
        return True
    elif e_x > 352 and p_x > 352:
        return True
    elif 352 > e_y > -352 and 352 > e_x > -352:
        if 352 > p_y > -352 and 352 > p_x > -352:
            return True
    return in_center(e, p)


def in_door(e, p):
    angle = p['angle']

    if tracker2(e) != 6 and tracker2(p) != 6:
        return True

    elif tracker2(e) == 6 or tracker2(p) == 6:
        if angle == 0.0 or angle == 90.0 or angle == 180 or angle == 270:
            return True
    return False


def tracker(t):
    coord = tracker2(t)
    if coord < 6:
        return coord
    else:
        return 1  # 6#"d"


def tracker2(t):
    t_x = t['x_position']
    t_y = t['y_position']
    if 318 > t_y > -318 and 318 > t_x > -318: # central
        return 1  # "c"
    elif t_y > 386: #north
        return 2  # "n"
    elif t_y < -386: #south
        return 3  # "s"
    elif t_x > 386: #east
        return 4  # "e"
    elif t_x < -386: #west
        return 5  # "w"
    else:
        return 6  # "d"


def target_in_room(enemies, p_coord):
    if p_coord == 6:
        return False

    for e in enemies:
        e_coor = tracker(e)
        if p_coord == e_coor:
            return True
    return False


def navigate(enemies, p_coord, player):
    if p_coord != 1:
        return 1

    if len(enemies) <= 0:
        return 0
    min_dist = 10000
    target = None
    for e in enemies:
        dist = get_dist(player, e)
        if min_dist > dist:
            min_dist = dist
            target = e

    return tracker(target)


def get_dist(player, enemy):
    pl_x = player['x_position']
    pl_y = player['y_position']

    en_x = enemy['x_position']
    en_y = enemy['y_position']
    e_coor = [en_x, en_y]
    p_coor = [pl_x, pl_y]

    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p_coor, e_coor)))  # math.dist(p_coor, e_coor)


# Utility function for getting angle from B-direction to A


def get_angle(target, start, offset):
    pl_x = target['x_position']
    pl_y = target['y_position']

    en_x = start['x_position']
    en_y = start['y_position']
    ang = start['angle'] + offset
    if ang < 0:
        ang = 360 + ang
    elif ang > 360:
        ang = ang - 360
    elif ang == 360:
        ang = 0

    en_ori = ang * 2 * np.pi / 360

    # Get angle between player and enemy
    # Convert enemy ori to unit vector
    v1_x = np.cos(en_ori)
    v1_y = np.sin(en_ori)

    enemy_vector = np.asarray([v1_x, v1_y]) / np.linalg.norm(np.asarray([v1_x, v1_y]))

    # If its buggy throw random value out
    if np.linalg.norm(np.asarray([pl_x - en_x, pl_y - en_y])) == 0:
        return np.random.rand() * 3.14

    enemy_face_vector = np.asarray([pl_x - en_x, pl_y - en_y]) / np.linalg.norm(
        np.asarray([pl_x - en_x, pl_y - en_y]))

    angle = np.arccos(np.clip(np.dot(enemy_vector, enemy_face_vector), -1.0, 1.0))
    
    sign = np.sign(np.linalg.det(np.stack((enemy_vector[-2:], enemy_face_vector[-2:]))))
    return angle, sign

