import mdptoolbox
import numpy as _np

action = ['north', 'south', 'east', 'west', 'get', 'put']

class PositionAction:
    north = True
    south = True
    west = True
    east = True

    is_term = False

class Map:
    SIZE = 5
    ACTION = 4

    def __init__(self, size):
        PositionAction pa
        self._map = [[pa for i in range(SIZE)] for j in range(SIZE)]

        # build outer wall
        for i in range(SIZE):
            self._map[0, SIZE - 1].east = False

        # build map with obstacles
        wall_length = (SIZE - 1) / 2
        for i in range(wall_length):
            self._map[i, (SIZE - 1) / 2].east = False
            self._map[i, (SIZE - 1) / 2].west = False

            self._map(SIZE - 1 - i, 1).east = False
            self._map(SIZE - 1 - i, 1).west = False

            self._map(SIZE - 1 - i, SIZE - 2).east = False
            self._map(SIZE - 1 - i, SIZE - 2).west = False

        # add terminal states
        self._map[0][0].is_term = True
        # self._map(SIZE - 1, 0).is_term = True
        # self._map(0, SIZE - 1).is_term = True
        # self._map(SIZE - 1, SIZE - 1).is_term = True

    def get_neighbor_index(self, i):
        x = int(i / SIZE)
        y = i - x*SIZE

        # in sequence: 'north', 'south', 'east', 'west'
        result = [(x-1)*SIZE + y, (x+1)*SIZE + y, x*SIZE + y-1, x*SIZE + y+1]

        # change validation
        if !self._map[x][y].north:
            result[0] = i
        if !self._map[x][y].south:
            result[1] = i
        if !self._map[x][y].east:
            result[2] = i
        if !self._map[x][y].west:
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
        p_wrong_dir = 0.1
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
            if result[i] != term_state
                R[result[i], get_opposite_action(i)] = 1

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

        for x in range(SIZE):
            for y in range(SIZE):
                current_action = policy_map[x][y]
                if current_action == 0:
                    print '^'
                elif current_action == 1:
                    print 'v'
                elif current_action == 2:
                    print '>'
                elif current_action == 3:
                    print '<'
            print '\n'

        
def __main__():
    Map m
    m.solve()
    m.display()
