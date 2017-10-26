import mdptoolbox
import numpy as _np
import random
import sys
from bitstring import BitArray

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

    def set_term(self, term):
        for x in range(SIZE):
            for y in range(SIZE):
                self._map[x][y].is_term = False
        self._map[term[0]][term[1]].is_term = True

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
        value = vi.V
        self.value_map = [[value[x*SIZE + y] for y in range(SIZE)] for x in range(SIZE)]

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

    def display_value(self):
        for x in range(SIZE):
            for y in range(SIZE):
                sys.stdout.write('%.2f ' % self.value_map[x][y])
            sys.stdout.write('\n');
        sys.stdout.write('\n')

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
    def __init__(self, encode):
        self.encode = encode
        self.locatiton = (0, 0)
        self.edge_list = []
        self.id = -1

class MDPEdge:
    def __init__(self, pid, cid, start, mid, term):
        self.parent_id = pid
        self.child_id = cid
        self.map = Map()
        self.start = start
        self.mid = mid
        self.term = term
        self.reward = 0

    def get_reward(self):
        if self.reward == -1:
            return -1
        self.map.set_term(self.mid)
        self.map.solve()
        value_map = self.map.value_map
        r1 = value_map[self.start[0]][self.start[1]]

        self.map.set_term(self.term)
        self.map.solve()
        value_map = self.map.value_map
        r2 = value_map[self.mid[0]][self.mid[1]]
        self.reward = r1 + r2 
        return self.reward

class AMDP:
    def __init__(self):
        self.map = Map()
        self.code_list = []
        self.node_list = []
        self.pas_count = 3
        self.root = None
        self.pas_start = [(0, 0), (0, 14), (14, 0)]
        self.pas_end = [(14, 14), (14, 0), (0, 0)]

    def create_child(self, parent):
        if parent.encode.uint == (2**self.pas_count - 1):
            child_term = BitArray(uint = (2**self.pas_count-1), length = self.pas_count)
            self.node_list += [MDPNode(child_term)]
            return
        for i in range(self.pas_count):
            if parent.encode.bin[i] == '1':
                child = MDPNode(parent.encode)
                child.id = parent.id 
                child.location = parent.location
                edge = MDPEdge(parent.id, child.id, parent.location, parent.location, child.location)
                edge.reward = -1
                parent.edge_list += [edge]
                continue
                
            child_code = BitArray(parent.encode)
            child_code[i] = 1
            child= MDPNode(child_code)
            child.id = len(self.node_list)
            child.location = self.pas_end[i]
            edge = MDPEdge(parent.id, child.id, parent.location, self.pas_start[i], child.location)
            parent.edge_list += [edge]
            self.node_list += [child]
            self.create_child(child)

    def build_graph(self):
        taxi_start = (5, 5)
        pas_count = self.pas_count
        for i in range(2**pas_count):
            encode = BitArray(uint=i, length=pas_count)
            self.code_list += [encode]
        
        # build the graph
        self.root = MDPNode(self.code_list[0])
        self.root.location = taxi_start
        self.root.id = 0
        self.node_list += [self.root]
        # create children dfs
        self.create_child(self.root)

    def generate_mdp(self):
        total_state = len(self.node_list)
        T = _np.zeros((self.pas_count, total_state, total_state))
        for i in range(total_state):
            for a in range(self.pas_count):
                if len(self.node_list[i].edge_list) == 0:
                    T[a, i, i] = 1
                    continue
                cid = self.node_list[i].edge_list[a].child_id
                T[a, i, cid] = 1
        self.T = T
        
        # build reward matrix
        R = _np.zeros((total_state, self.pas_count))
        for i in range(total_state):
            for a in range(self.pas_count):
                if len(self.node_list[i].edge_list) == 0:
                    continue
                edge = self.node_list[i].edge_list[a] 
                if edge.reward == -1:
                    continue
                R[i, a] = edge.get_reward()
        self.R = R

    def solve(self):
        vi = mdptoolbox.mdp.ValueIteration(self.T, self.R, 0.95)
        vi.run()

        policy = vi.policy
        self.policy = policy

    def display(self):
        print self.policy

        
if __name__ == '__main__':
    amdp = AMDP()
    amdp.build_graph()
    amdp.generate_mdp()
    amdp.solve()
    amdp.display()
