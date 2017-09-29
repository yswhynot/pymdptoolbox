import mdptoolbox
import numpy as _np
import random
import sys

action = ['north', 'south', 'east', 'west', 'get', 'put', 'root']

SIZE = 15
ACTION = 4

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

    def generate_mdp(self):
        p_wrong_dir = 0.2/3
        p_dir = 1 - 3*p_wrong_dir
        total_state = SIZE*SIZE

        # Transition matrix
        T = _np.zeros((ACTION, total_state, total_state))
        for i in range(total_state):
            neighbor = self.get_neighbor_index(i)

            for j in range(ACTION):
                for k in range(ACTION):
                    if j == k:
                        T[j, i, neighbor[k]] += p_dir
                    else:
                        T[j, i, neighbor[k]] += p_wrong_dir
        # print T[0, :, :]

        # Reward matrix
        R = _np.zeros((total_state, ACTION))
        # find the termination state
        term_state = -1
        for x in range(SIZE):
            for y in range(SIZE):
                if self._map[x][y].is_term:
                    term_state = x*SIZE + y
                    break
            if term_state != -1:
                break
        # update neighbor to termination state reward
        result = self.get_neighbor_index(term_state)
        for i in range(ACTION):
            if result[i] != term_state:
                R[result[i], self.get_opposite_action(i)] = 1

        return (T, R)

    def solve(self):
        T, R = self.generate_mdp()
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.95)
        vi.run()

        # parse policy
        policy = vi.policy
        self.policy_map = [[policy[x*SIZE + y] for y in range(SIZE)] for x in range(SIZE)]

    def display(self):
        policy_map = self.policy_map

        sys.stdout.write("\n")
        for x in range(SIZE):
            for y in range(SIZE):
                current_action = policy_map[x][y]
                if current_action == 0:
                    sys.stdout.write('^')
                elif current_action == 1:
                    sys.stdout.write('v')
                elif current_action == 2:
                    sys.stdout.write('>')
                elif current_action == 3:
                    sys.stdout.write('<')
                sys.stdout.write(' ')
            sys.stdout.write('\n')
        # self.print_map(2)

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
