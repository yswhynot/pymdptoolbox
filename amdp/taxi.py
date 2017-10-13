import mdptoolbox
import numpy as _np
import random
import sys
import copy
import time

action = ['north', 'south', 'east', 'west', 'get', 'put', 'root']

SIZE = 5
ACTION = 5
NUM = 2

class PositionAction:
    north = True
    south = True
    west = True
    east = True

    is_term = False

class Map:

    def __init__(self):
        self._map = [[PositionAction() for i in range(SIZE)] for j in range(SIZE)]

        # build outer wall
        for i in range(SIZE):
            self._map[0][i].north = False
            self._map[i][0].west = False
            self._map[SIZE - 1][i].south = False
            self._map[i][SIZE - 1].east = False

        # build map with obstacles
        wall_length = (SIZE - 1) / 2
        for i in range(wall_length):
            self._map[i][(SIZE - 1) / 2].east = False
            self._map[i][(SIZE - 1) / 2].west = False

            self._map[SIZE - 1 - i][2].east = False
            self._map[SIZE - 1 - i][2].west = False

            self._map[SIZE - 1 - i][SIZE - 3].east = False
            self._map[SIZE - 1 - i][SIZE - 3].west = False

        # add terminal states
        # self._map[0][0].is_term = True
        # self._map(SIZE - 1, 0).is_term = True
        # self._map(0, SIZE - 1).is_term = True
        # self._map(SIZE - 1, SIZE - 1).is_term = True

    def set_state(self, state):
        if state == 'get':
            self._map[0][0].is_term = True
        elif state == 'put':
            rand = random.uniform(0, 1)
            if rand < 0.7:
                self._map[SIZE-1][SIZE-1].is_term = True
            else:
                self._map[0][SIZE-1].is_term = True

    def get_neighbor_index(self, i):
        x = int(i / SIZE)
        y = i - x*SIZE

        # in sequence: 'north', 'south', 'east', 'west'
        result = [(x-1)*SIZE + y, (x+1)*SIZE + y, x*SIZE + y+1, x*SIZE + y-1]

        # wrap boulder
        if result[2] == (SIZE*SIZE):
            result[2] -= 1

        # change validation
        if not self._map[x][y].north:
            result[0] = i
        if not self._map[x][y].south:
            result[1] = i
        if not self._map[x][y].east:
            result[2] = i
        if not self._map[x][y].west:
            result[3] = i

        return result

    def get_opposite_action(self, a):
        if a == 0:
            return 1
        if a == 1:
            return 0
        if a == 2:
            return 3
        if a == 3:
            return 2
        if a == 4:
            return 4

    def get_neighbor(self, x, y, action):
        result = (x, y)
        if action == 0:
            if self._map[x][y].north:
                result = (x-1, y)
        elif action == 1:
            if self._map[x][y].south:
                result = (x+1, y)
        elif action == 2:
            if self._map[x][y].east:
                result = (x, y+1)
        elif action == 3:
            if self._map[x][y].west:
                result = (x, y-1)
        elif action == 4:
            result = (x, y)
        return result

    def get_joint_neighbor(self, s1, s2, a1, a2):
        result = [s1, s2]
        n1 = self.get_neighbor(s1[0], s1[1], a1)
        n2 = self.get_neighbor(s2[0], s2[1], a2)
        if n1 == n2:
            result = [n1, s2]
        else:
            result = [n1, n2]
        return result


    def generate_mdp(self):
        p_wrong_dir = 0.2/4
        p_dir = 1 - 4*p_wrong_dir
        total_state = SIZE*SIZE*SIZE*SIZE

        # Transition matrix
        T = _np.zeros((ACTION*ACTION, total_state, total_state))
        self.states = []
        for x1 in range(SIZE):
            for y1 in range(SIZE):
                for x2 in range(SIZE):
                    for y2 in range(SIZE):
                        self.states += [(x1, y1, x2, y2)]
        for state in self.states:
            (x1, y1, x2, y2) = state
            current_state = x1*SIZE*SIZE*SIZE + y1*SIZE*SIZE + x2*SIZE + y2
            for a1 in range(ACTION):
                for a2 in range(ACTION):
                    for k1 in range(ACTION):
                        for k2 in range(ACTION):
                            r = self.get_joint_neighbor((x1, y1), (x2, y2), k1, k2)
                            index_r = r[0][0]*SIZE*SIZE*SIZE + r[0][1]*SIZE*SIZE + r[1][0]*SIZE + r[1][1]
                            if a1 == k1 and a2 == k2:
                                T[a1*ACTION + a2, current_state, index_r] += p_dir*p_dir
                            elif a1 != k1 and a2 != k2:
                                T[a1*ACTION + a2, current_state, index_r] += p_wrong_dir * p_wrong_dir
                            else:
                                T[a1*ACTION + a2, current_state, index_r] += p_dir * p_wrong_dir 

        #  print T[1, 0, :]

        # Reward matrix
        R = _np.zeros((total_state, ACTION*ACTION))
        # find the termination state
        term_state = []
        for state in self.states:
            (x1, y1, x2, y2) = state
            if self._map[x1][y1].is_term or self._map[x2][y2].is_term:
                term_state += [(x1, y1, x2, y2)] 

        # update neighbor to termination state reward
        for state in term_state:
            (x1, y1, x2, y2) = state
            for a1 in range(ACTION):
                for a2 in range(ACTION):
                    r = self.get_joint_neighbor((x1, y1), (x2, y2), a1, a2)
                    index_r = r[0][0]*SIZE*SIZE*SIZE + r[0][1]*SIZE*SIZE + r[1][0]*SIZE + r[1][1]
                    oppo_action = self.get_opposite_action(a1)*ACTION+ self.get_opposite_action(a2)
                    R[index_r, oppo_action] = 1
        return (T, R)

    def solve(self):
        T, R = self.generate_mdp()
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.95)
        vi.run()

        # parse policy
        policy = vi.policy
        self.policy_map = [[[[policy[x1*SIZE*SIZE*SIZE + y1*SIZE*SIZE + x2*SIZE + y2] for y2 in range(SIZE)] for x2 in range(SIZE)] for y1 in range(SIZE)] for x1 in range(SIZE)]

    def get_policy_char(self, a):
        if a == 0:
            return '^'
        elif a == 1:
            return 'v'
        elif a == 2:
            return '>'
        elif a == 3:
            return '<'
        elif a == 4:
            return '.'

    def display_action(self, a1, a2):
        for x1 in range(SIZE):
            for y1 in range(SIZE):
                print "a1: (%d, %d)" % (x1, y1)
                for x2 in range(SIZE):
                    print a1[x1][y1][x2]
                print "a2:"
                for x2 in range(SIZE):
                    print a2[x1][y1][x2]

    def display(self):
        policy_map1 = copy.deepcopy(self.policy_map)
        policy_map2 = copy.deepcopy(self.policy_map)

        for state in self.states:
            (x1, y1, x2, y2) = state
            a2 = self.policy_map[x1][y1][x2][y2] % ACTION
            a1 = (self.policy_map[x1][y1][x2][y2] - a2) / ACTION
            policy_map1[x2][y2][x1][y1] = self.get_policy_char(a1)
            policy_map2[x1][y1][x2][y2] = self.get_policy_char(a2)
        self.display_action(policy_map1, policy_map2)

    def print_map(self, action):
        pmap = self._map

        for x in range(SIZE):
            for y in range(SIZE):
                if action == 0:
                    if pmap[x][y].north:
                        sys.stdout.write('o')
                    else:
                        sys.stdout.write('1')
                if action == 1:
                    if pmap[x][y].south:
                        sys.stdout.write('o')
                    else:
                        sys.stdout.write('1')
                if action == 2:
                    if pmap[x][y].east:
                        sys.stdout.write('o')
                    else:
                        sys.stdout.write('1')
                if action == 3:
                    if pmap[x][y].west:
                        sys.stdout.write('o')
                    else:
                        sys.stdout.write('1')
                sys.stdout.write(' ')
            sys.stdout.write('\n')

class MDPNode:
    def __init__(self, name, parent):
        self.name = name
        self.childs = []

        if parent is '':
            return
        parent.add_child(self)
        self.parents = [parent]

    def add_child(self, child):
        self.childs.append(child)

    def add_parent(self, parent):
        self.parents.append(parent)

    def solve(self):
        for child in self.childs:
            child.solve()

    def display(self):
        sys.stdout.write("\n")
        sys.stdout.write(self.name)
        for child in self.childs:
            child.display()

class AMDP:
    def __init__(self):
        self.root = MDPNode('root', '')
        get_node = MDPNode('get', self.root)
        put_node = MDPNode('put', self.root)
        pick_node = MDPNode('pick', get_node)
        drop_node = MDPNode('drop', put_node)
        self.nav_node = MDPNode('nav', put_node)
        self.nav_node.add_parent(get_node)
        
        self.mdp_map = Map()
        self.nav_node.add_child(self.mdp_map)

    def solve(self, action):
        self.mdp_map.set_state(action)
        self.root.solve()

    def display(self):
        self.root.display()

        
if __name__ == '__main__':
    amdp = AMDP()
    amdp.solve(sys.argv[1])
    amdp.display()
