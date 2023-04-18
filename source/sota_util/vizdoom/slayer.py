import torch
from sota_util.vizdoom.doom_net import Net as net
from sota_util.vizdoom.myutil import v_wrap, tracker, target_sighted, get_dist, target_in_room, navigate, \
    gunner, in_center2, in_door, in_center3, h_check, to_center, to_border, ob_help, tracker2, get_angle
import numpy as np


class Balance:

    def __init__(self):
        self.possible_answers = list()
        self.possible_answers.append(dict({'action': 'right'}))
        self.possible_answers.append(dict({'action': 'left'}))
        self.possible_answers.append(dict({'action': 'forward'}))
        self.possible_answers.append(dict({'action': 'backward'}))
        self.possible_answers.append(dict({'action': 'nothing'}))
        # Cartpole Agent
        #self.agent = None
        self.agent = net(25, 5, 128, 64)
        self.agent.load_state_dict(torch.load("sota_util/vizdoom/balance.txt"))
        return

    def predict(self, feature_vector):
        state_vec = self.breaker(feature_vector)
        act = self.agent.choose_action(v_wrap(state_vec[None, :]))
        label_prediction = self.possible_answers[act]
        return label_prediction

    def break_cart(self, cart):
        temp = []

        temp.append(float(cart['x_position']))
        temp.append(float(cart['y_position']))
        temp.append(float(cart['z_position']))
        temp.append(float(cart['x_velocity']))
        temp.append(float(cart['y_velocity']))
        temp.append(float(cart['z_velocity']))
        return temp

    def break_pole(self, pole):
        temp = []
        temp.append(float(pole['x_quaternion']))
        temp.append(float(pole['y_quaternion']))
        temp.append(float(pole['z_quaternion']))
        temp.append(float(pole['w_quaternion']))
        temp.append(float(pole['x_velocity']))
        temp.append(float(pole['y_velocity']))
        temp.append(float(pole['z_velocity']))
        return temp

    def break_blocks(self, blocks):
        temp = []
        for block in blocks:
            if len(temp) < 12:
                temp.append(float(block['x_position']))
                temp.append(float(block['y_position']))
                temp.append(float(block['z_position']))

        return temp

    def breaker(self, state):
        cart = self.break_cart(state['cart'])
        pole = self.break_pole(state['pole'])
        blocks = self.break_blocks(state['blocks'])
        vec = cart + pole + blocks

        while len(vec) < 25:
            vec.append(0.0)

        if len(vec) > 25:
            vec = vec[0:25]

        return np.asarray(vec)


class Vincent:

    def __init__(self):
        self.possible_answers = list()
        self.possible_answers.append(dict({'action': 'nothing'}))
        self.possible_answers.append(dict({'action': 'left'}))
        self.possible_answers.append(dict({'action': 'right'}))
        self.possible_answers.append(dict({'action': 'forward'}))
        self.possible_answers.append(dict({'action': 'backward'}))
        self.possible_answers.append(dict({'action': 'shoot'}))
        self.possible_answers.append(dict({'action': 'turn_left'}))
        self.possible_answers.append(dict({'action': 'turn_right'}))

        self.gnav = None
        #self.gchar = None
        self.gammo = None

        self.step = 0
        self.override = False
        self.pl_x = 0  # previous move x coord
        self.pl_y = 0  # previous move y coord
        self.navigation = False
        self.resupply = False  # find health or ammo
        self.r_load = False
        #self.traveling = False
        self.medic = False
        self.manual = False  # safeguard for if the navigation agent gets stuck
        self.fired = False
        self.pact = 0  # previous action
        self.e_count = 0
        self.i_targ = 0
        self.e_list = {}

        self.actions = [1, 2, 4, 3, 6, 7]

        self.c_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 468.0},
                       {"x_position": 0.0, "y_position": -468.0}, {"x_position": 468.0, "y_position": 0.0},
                       {"x_position": -468.0, "y_position": 0.0}]
        self.s_list = [{"x_position": 180.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 180.0},
                       {"x_position": -180.0, "y_position": 0}, {"x_position": 0, "y_position": -180.0}]

        return

    def predict(self, feature_vector):
        return self.slayer_agent(feature_vector=feature_vector)

    # named after the main character of doom
    def slayer_agent(self, feature_vector: dict) -> dict:
        """Process an episode data instance.
        Parameters
        ----------
        feature_vector : dict
            The dictionary of the feature vector.  Domain specific feature vector formats are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """

        found_error = False
        # Check for some error conditions and provide helpful feedback.
        if 'enemies' not in feature_vector:
            self.log.error('This does not appear to be a vizdoom feature vector! Did you forget '
                           'to set the correct domain in the agent config file?')
            found_error = True

        player = feature_vector['player']
        self.step += 1
        pl_x2 = player['x_position']
        pl_y2 = player['y_position']
        stuck = False
        hit = False

        health = int(player['health'])
        items = feature_vector['items']
        ammo = int(player['ammo'])
        p_coord = tracker(player)
        if ammo < 1 and len(items['ammo']) <= 0:  # end the round quickly
            return self.possible_answers[0]

        nav_vec, e_temp, combat, clip, med, ob_list, elist = self.breaker(feature_vector)

        check = True
        if not in_center2(player) and combat == 0:
            obst = items['obstacle']
            for o in obst:
                if get_dist(o, player) < 80:
                    check = False

        if ammo >= 3 and self.r_load:
            self.r_load = False
            self.resupply = False

        if health > 30 and self.medic:
            self.resupply = False
            self.medic = False

        if int(self.pl_x) == int(pl_x2) and int(self.pl_y) == int(pl_y2):
            if self.pact == 0 or self.pact == 1 or self.pact == 2 or self.pact == 3:
                stuck = True
                if self.navigation and ammo > 0:
                    if self.resupply:
                        self.manual = True
                        self.medic = False
                        self.r_load = False

        if len(elist) > len(self.e_list):
            self.e_list = elist

        for key in elist.keys():

            if elist[key] < self.e_list[key]:
                if self.fired:
                    hit = True

        if self.fired and not hit:  # prevents out of control spam firing
            self.manual = False
        self.e_list = elist

        self.pl_x = pl_x2
        self.pl_y = pl_y2

        self.e_count = e_temp

        label_prediction = self.possible_answers[5]
        act = np.dtype('int64').type(4)  # shooting is default
        self.fired = False

        self.navigation = False
        if ammo < 1:
            self.manual = False

        if get_dist(player, self.s_list[self.i_targ]) <= 60.0:
            self.i_targ += 1
            if self.i_targ == 4:
                self.i_targ = 0
        tir = target_in_room(feature_vector['enemies'], p_coord)
        if combat > 0 or tracker2(player) != 6:
            self.override = False

        if combat == 1 and ammo > 0:  # essentially a smart gun that forces the agent to fire or turn towards easy targets
            self.fired = True
            self.navigation = True

        elif combat == 2 and ammo > 0:
            act = np.dtype('int64').type(5)
            label_prediction = self.possible_answers[6]
            self.navigation = True

        elif combat == 3 and ammo > 0:
            act = np.dtype('int64').type(6)
            label_prediction = self.possible_answers[7]
            self.navigation = True
        elif combat == 4 and ammo > 0:
            act = np.dtype('int64').type(0)
            label_prediction = self.possible_answers[1]

        elif combat == 5 and ammo > 0:
            act = np.dtype('int64').type(1)
            label_prediction = self.possible_answers[2]
        else:
            if stuck:
                act = self.pact
                while act == self.pact:
                    act = np.dtype('int64').type(np.random.randint(0, 3))

                label_prediction = self.possible_answers[self.actions[act]]
            elif self.override and tracker2(player) == 6:
                act = np.dtype('int64').type(3)
                label_prediction = self.possible_answers[self.actions[act]]

            elif tir and (ammo > 0 or len(items['ammo']) <= 0):  # and not resupply:
                act = np.dtype('int64').type(0)
                if player['angle'] != 90:
                    if 270 > player['angle'] > 90:
                        act = np.dtype('int64').type(5)  # 'turn_right'
                    else:
                        act = np.dtype('int64').type(4)
                elif p_coord == 1:

                    nav_vec[4] = self.s_list[self.i_targ]['x_position']
                    nav_vec[5] = self.s_list[self.i_targ]['y_position']
                    nav_vec[6] = get_dist(player, self.s_list[self.i_targ])

                    act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    nav_vec[4] = self.c_list[p_coord - 1]['x_position']
                    nav_vec[5] = self.c_list[p_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.c_list[p_coord - 1])
                    act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))
                label_prediction = self.possible_answers[self.actions[act]]
            else:  # navigation net
                act = np.dtype('int64').type(0)
                if med and ((health <= 25) or (get_dist(med, player) < 250 and tracker(
                        med) == p_coord and not tir and self.step < 1001)):  # health <= 30 and len(items['health']) > 0 and not self.manual and not tir:
                    m_coord = tracker(med)
                    if m_coord == p_coord:

                        self.resupply = True
                        self.medic = True

                        if player['angle'] == 90:
                            nav_vec[4] = float(med['x_position'])
                            nav_vec[5] = float(med['y_position'])
                            nav_vec[6] = get_dist(med, player)
                            act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))

                        else:

                            if 270 > player['angle'] > 90:
                                act = np.dtype('int64').type(5)  # 'turn_right' bookmark
                            else:
                                act = np.dtype('int64').type(4)


                    elif p_coord == 1:
                        targ_coord = m_coord
                        if not in_center3(player):  # doors
                            act = to_center(player, m_coord)  # door_man2(player, m_coord)
                        elif player['angle'] != 315 and check:
                            if 315 > player['angle'] > 135:
                                act = np.dtype('int64').type(4)  # 'turn_right'
                            else:
                                act = np.dtype('int64').type(5)
                        else:
                            nav_vec[4] = self.c_list[m_coord - 1]['x_position']
                            nav_vec[5] = self.c_list[m_coord - 1]['y_position']
                            nav_vec[6] = get_dist(player, self.c_list[m_coord - 1])
                            act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))
                    else:
                        act = to_center(player, p_coord)
                        self.override = True





                elif clip and (ammo < 1 or (ammo < 3 and (get_dist(clip, player) < 250 and tracker(
                        clip) == p_coord and not tir and self.step < 1001))):  # ammo < 1 and len(items['ammo']) > 0 and not self.manual:
                    a_coord = tracker(clip)
                    if a_coord == p_coord:
                        if player['angle'] == 90:
                            nav_vec[4] = float(clip['x_position'])
                            nav_vec[5] = float(clip['y_position'])
                            nav_vec[6] = get_dist(player, clip)
                            act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))

                        else:
                            if 270 > player['angle'] > 90:
                                act = np.dtype('int64').type(5)  # 'turn_right'
                            else:
                                act = np.dtype('int64').type(4)
                        self.r_load = True
                        self.resupply = True


                    elif p_coord == 1:
                        targ_coord = a_coord
                        if not in_center3(player) and check:  # doors
                            act = to_center(player, a_coord)  # door_man2(player, a_coord)

                        elif player['angle'] != 315:
                            if 315 > player['angle'] > 135:
                                act = np.dtype('int64').type(4)  # 'turn_left'
                            else:
                                act = np.dtype('int64').type(5)
                        else:
                            nav_vec[4] = self.c_list[a_coord - 1]['x_position']
                            nav_vec[5] = self.c_list[a_coord - 1]['y_position']
                            nav_vec[6] = get_dist(player, self.c_list[a_coord - 1])
                            act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))

                    else:

                        act = to_center(player, p_coord)
                        self.override = True

                else:
                    if p_coord != 1:

                        act = to_border(player, p_coord)  # corner3(player, p_coord)
                    else:
                        targ_coord = navigate(feature_vector['enemies'], p_coord, player)

                        if not in_center3(player) and check:  # doors
                            act = to_center(player, targ_coord)  # door_man2(player, targ_coord)
                            self.override = True
                        elif player['angle'] != 315:
                            
                            if 315 > player['angle'] > 135:
                                act = np.dtype('int64').type(4)  # turn_left
                            else:
                                act = np.dtype('int64').type(5)

                        else:
                            nav_vec[4] = self.c_list[targ_coord - 1]['x_position']
                            nav_vec[5] = self.c_list[targ_coord - 1]['y_position']
                            nav_vec[6] = get_dist(player, self.c_list[targ_coord - 1])
                            act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))

                # self.traveling = False
                self.navigation = True
                # act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))
                label_prediction = self.possible_answers[self.actions[act]]

        self.pact = act

        return label_prediction

    def reset_vars(self):
        self.pl_x = 0
        self.pl_y = 0

        self.pact = 0
        #self.traveling = False
        self.navigation = False
        self.resupply = False  # find health or ammo
        self.r_load = False
        self.medic = False
        self.manual = False
        self.fired = False
        self.e_list = {}
        self.e_count = 0
        self.i_targ = 0
        self.step = 0
        self.override = False


    def reset_slayer(self):
        self.gnav = None
        #self.gchar = None
        self.gnav = net(13, 6, 64, 32)
        self.gnav.load_state_dict(torch.load("sota_util/vizdoom/navigation_4.txt"))
        #self.gchar= net(20, 6, 64, 32)
        #self.gchar.load_state_dict(torch.load("sota_util/vizdoom/charge_3.txt"))
        self.gammo = net(13, 6, 64, 32)
        self.gammo.load_state_dict(torch.load("sota_util/vizdoom/ammo_4.txt"))

    def break_health(self, items, player, p_coord, enemies, o_list):
        if len(items['health']) <= 0:
            return None
        min_dist = 10000

        m_health = None
        med_list = items['health']
        for h in med_list:
            dist = get_dist(player, h)
            if min_dist > dist:
                safe = True
                for o in o_list:

                    if get_dist(o, h) < 40:
                        safe = False
                        break

                if safe:
                    min_dist = dist
                    m_health = h
        if not m_health:
            return None
        m_coord = tracker(m_health)
        if m_coord != p_coord:
            for h in med_list:
                h_c = tracker(h)
                if h_c == p_coord:
                    m_health = h
                    m_coord = h_c
                    break

        if m_coord != 1 and p_coord == m_coord:
            hazard = False
            for e in enemies:
                if tracker(e) == m_coord:
                    hazard = True
                    break
            if hazard:
                for h in med_list:
                    if tracker(h) == 1:
                        m_health = h
                        break

        return m_health

    def break_ammo(self, items, player, p_coord, enemies, o_list):
        clip_list = items['ammo']
        if len(clip_list) <= 0:
            return None
        min_dist = 10000

        m_ammo = None

        for a in clip_list:
            dist = get_dist(player, a)
            if min_dist > dist:
                safe = True
                for o in o_list:

                    if get_dist(o, a) < 40:
                        safe = False
                        break

                if safe:
                    min_dist = dist
                    m_ammo = a
        if not m_ammo:
            return None
        if len(enemies) > 0:
            a_coord = tracker(m_ammo)
            if a_coord != p_coord:
                for a in clip_list:
                    a_c = tracker(a)
                    if a_c == p_coord:
                        m_ammo = a
                        a_coord = a_c
                        break

            if a_coord != 1 and p_coord == a_coord:
                hazard = False
                for e in enemies:
                    if tracker(e) == a_coord:
                        hazard = True
                        break
                if hazard:
                    for a in clip_list:
                        if tracker(a) == 1:
                            m_ammo = a
                            break
        return m_ammo

    def break_traps(self, items, player):
        nav_trap = []
        if len(items['trap']) <= 0:
            return [0.0, 0.0, -1.0]

        min_dist = 10000
        m_trap = items['trap'][0]
        for t in items['trap']:
            dist = get_dist(player, t)
            if min_dist > dist:
                min_dist = dist
                m_trap = t

        nav_trap.append(float(m_trap['x_position']))
        nav_trap.append(float(m_trap['y_position']))
        nav_trap.append(min_dist)

        return nav_trap

    def break_obstacles(self, items, player):
        nav_obst = []
        ob_list = []
        min_dist = 10000
        m_obst = None

        for o in items['obstacle']:
            dist = get_dist(player, o)

            if target_sighted(o, player):
                ob_list.append(o)

            if min_dist > dist:
                min_dist = dist
                m_obst = o

        if len(items['obstacle']) <= 0:
            nav_obst = [0.0, 0.0, 0.0]

        else:
            nav_obst.append(float(m_obst['x_position']))
            nav_obst.append(float(m_obst['y_position']))
            nav_obst.append(min_dist)

        return nav_obst, ob_list

    def break_enemy(self, enemies, player):
        nav_enemy = []
        elist = {}
        min_dist = 10000
        m_enemy = None
        for e in enemies:
            dist = get_dist(player, e)
            elist[int(e['id'])] = h_check(e)
            if min_dist > dist and target_sighted(e, player):
                min_dist = dist
                m_enemy = e

        if not m_enemy:
            nav_enemy = [0.0, 0.0, 10000.0]

        else:
            nav_enemy.append(float(m_enemy['x_position']))
            nav_enemy.append(float(m_enemy['y_position']))
            nav_enemy.append(min_dist)

        return nav_enemy, elist

    def breaker(self, state):
        enemies = state['enemies']
        items = state['items']
        player = state['player']

        a_1 = True
        a_2 = True
        a_3 = True
        a_4 = True

        combat = 0  # 0 nothing, 1 attack, 2 left, 3 right, 4 m left, 5 m right
        nav_enemy, elist = self.break_enemy(enemies, player)

        p_coord = tracker(player)
        nav_obst, ob_list = self.break_obstacles(items, player)
        e_count = len(enemies)
        nav_trap = self.break_traps(items, player)
        safe = True

        if nav_trap and nav_trap[2] < 60:
            safe = False

        w_0 = 0
        if tracker2(player) != 6:
            w_0 = 20
        if player['ammo'] > 0:
            for e in enemies:

                if target_sighted(e, player) and in_door(e, player):
                    dist = get_dist(player, e)
                    a_0 = False

                    if gunner(e, player, 0.0, 40 + w_0):

                        barr = ob_help(ob_list, player, dist, 0.0)
                        is_clear = True
                        if not safe:
                            is_clear = False
                            w_0 = 0

                        if barr:
                            if gunner(e, player, 0.0, 18):
                                combat = 1
                                break
                            elif is_clear:
                                ang, sign = get_angle(e, player, 0.0)

                                check3 = True
                                check4 = False
                                check5 = True
                                for o in ob_list:

                                    if get_dist(player, o) < 60:
                                        check4 = True
                                        ang2, sign2 = get_angle(o, player, 0.0)
                                        ang2 = ang2 * 180 / np.pi
                                        if 135 > ang2 > 45:
                                            if sign2 < 1:

                                                check3 = False
                                            else:
                                                check5 = False

                                if check3 or check5:

                                    if sign < 1 and check3:
                                        # print("right")
                                        combat = 5
                                        if not check4:
                                            a_0 = True  # we don't want to repeat
                                            a_2 = True
                                    elif sign == 1 and check5:
                                        # print("left")
                                        combat = 4
                                        if not check4:
                                            a_0 = True  # we don't want to repeat
                                            a_2 = True

                    if a_1:
                        w_1 = 30 + w_0
                        if a_0:
                            w_1 = 18

                        if gunner(e, player, 45.0, w_1) and ob_help(ob_list, player, dist, 45.0):
                            combat = 2
                            a_1 = False

                        if a_1 and gunner(e, player, -45.0, w_1) and ob_help(ob_list, player, dist, -45.0):
                            combat = 3
                            a_1 = False

                        if a_1 and a_2 and gunner(e, player, 90.0, 30 + w_0) and ob_help(ob_list, player, dist, 90.0):
                            combat = 2
                            a_2 = False
                        if a_1 and a_2 and gunner(e, player, -90.0, 30 + w_0) and ob_help(ob_list, player, dist, -90.0):
                            combat = 3
                            a_2 = False

                        if a_1 and a_2 and a_3 and gunner(e, player, 135.0, 30 + w_0) and ob_help(ob_list, player, dist,
                                                                                                  135.0):
                            combat = 2
                            a_3 = False

                        if a_1 and a_2 and a_3 and gunner(e, player, -135.0, 30 + w_0) and ob_help(ob_list, player,
                                                                                                   dist,
                                                                                                   -135.0):
                            combat = 3
                            a_3 = False

                        if a_1 and a_2 and a_3 and a_4 and gunner(e, player, 180.0, 30 + w_0) and ob_help(ob_list,
                                                                                                          player, dist,
                                                                                                          180.0):
                            combat = 2
                            a_4 = False

        clip = self.break_ammo(items, player, p_coord, enemies, ob_list)
        med = self.break_health(items, player, p_coord, enemies, ob_list)

        if nav_enemy[2] < 90.0 and (nav_enemy[2] < nav_trap[2] or nav_trap[2] <= 0):
            nav_trap = nav_enemy
        h = 30
        if player['health'] <= 25:
            h = 5
        avatar2 = [float(player['x_position']), float(player['y_position']), float(player['angle']), h]

        sensor_vec2 = avatar2 + [0.0, 0.0, 0.0] + nav_obst + nav_trap
        return np.asarray(sensor_vec2), e_count, combat, clip, med, ob_list, elist

