"""Implementation of enemies gridworld environment family.
"""

import itertools
from collections import OrderedDict
import numpy as np
from envs.rendering_utils import load_asset, render_from_layout
from envs.env_base import Environment, TransitionModel, RewardFunction
from structs import StateVariable, DiscreteVariable, StateFactory
from settings import GeneralConfig as gc
from settings import SolverConfig as sc


EASY, MEDIUM, HARD = None, None, None


class EnemiesTransitionModel(TransitionModel):
    """Transition model for the enemies environments.
    """
    def __init__(self, env):
        self._env = env
        self._action_var = self.get_action_var(self.get_state_vars())

    def get_state_vars(self):
        return self._env.state_variables

    def get_action_var(self, state_vars):
        # Action: [right, left, up, down; slay each enemy]
        num_enemies = sum([1 for v in state_vars if v.name.startswith("enemy")])
        action_var = DiscreteVariable("action", 4+num_enemies)
        return action_var

    def model(self, state, action):
        # Only use this for CSI learning. Too complicated in stochastic case.
        assert self._env.ENEMY_MOVE_PROB == 1.0
        next_state = self.sample_next_state(state, action)
        yield next_state, 1.0

    def sample_next_state(self, state, action):
        # Update robot room and position.
        room = state[self._env.robot_room]
        pos = state[self._env.robot_pos]
        room, pos = self._update_robot(room, pos, action)
        next_state_dict = state.todict()
        next_state_dict[self._env.robot_room] = room
        next_state_dict[self._env.robot_pos] = pos
        try:
            enemy_room_ind = self._env.enemy_rooms.index(room)+1
        except ValueError:
            enemy_room_ind = 0  # robot not in same room as an enemy
        next_state_dict[self._env.robot_enemy_room] = enemy_room_ind
        # Update enemy positions.
        for room in self._env.enemy_rooms:
            for enemy_pos in self._env.enemy_poses[room]:
                if gc.rand_state.rand() < self._env.ENEMY_MOVE_PROB:
                    next_state_dict[enemy_pos] = self._update_enemy(
                        state[enemy_pos])
        # Handle slaying enemies.
        if action >= 4:
            enemy_to_slay = action-4
            enemies = [v for v in self.get_state_vars()
                       if v.name.startswith("enemy")]
            enemy_pos = enemies[enemy_to_slay]
            if self._distance(state[self._env.robot_pos],
                              state[enemy_pos]) <= 3:
                if state[self._env.robot_pos] < (self._env.GRID_SIZE*
                                                 self._env.GRID_SIZE/2):
                    next_state_dict[enemy_pos] = (self._env.GRID_SIZE*
                                                  self._env.GRID_SIZE-1)
                else:
                    next_state_dict[enemy_pos] = 0
        # Handle crashing (teleport back to start).
        if state[self._env.robot_room] in self._env.enemy_poses:
            poses = self._env.enemy_poses[state[self._env.robot_room]]
            if any(next_state_dict[self._env.robot_pos] == next_state_dict[ene]
                   for ene in poses):
                next_state_dict[self._env.robot_room] = self._env.INIT[0]
                next_state_dict[self._env.robot_pos] = self._env.INIT[1]
        return state.state_factory.build(next_state_dict)

    def ml_next_state(self, state, action):
        old_prob = self._env.ENEMY_MOVE_PROB
        self._env.ENEMY_MOVE_PROB = 1.0
        next_state = self.sample_next_state(state, action)
        self._env.ENEMY_MOVE_PROB = old_prob
        return next_state

    def _distance(self, pos1, pos2):
        pos1_x, pos1_y = self._env.pos_ind_to_coords(pos1)
        pos2_x, pos2_y = self._env.pos_ind_to_coords(pos2)
        return abs(pos1_x-pos2_x)+abs(pos1_y-pos2_y)

    def _update_robot(self, room, pos, action):
        room_x, room_y = self._env.room_ind_to_coords(
            self._env.rev_layout[room])
        pos_x, pos_y = self._env.pos_ind_to_coords(pos)
        if action == 0:  # right
            if pos_y == self._env.GRID_SIZE-1:  # change rooms
                if room_y == self._env.SUPERGRID_SIZE-1:
                    return room, pos  # no room -> no change
                room = self._env.layout[self._env.room_coords_to_ind(
                    room_x, room_y+1)]
                pos = self._env.pos_coords_to_ind(pos_x, 0)
                return room, pos
            pos = self._env.pos_coords_to_ind(pos_x, pos_y+1)
            return room, pos
        if action == 1:  # left
            if pos_y == 0:  # change rooms
                if room_y == 0:
                    return room, pos  # no room -> no change
                room = self._env.layout[self._env.room_coords_to_ind(
                    room_x, room_y-1)]
                pos = self._env.pos_coords_to_ind(pos_x, self._env.GRID_SIZE-1)
                return room, pos
            pos = self._env.pos_coords_to_ind(pos_x, pos_y-1)
            return room, pos
        if action == 2:  # up
            if pos_x == 0:  # change rooms
                if room_x == 0:
                    return room, pos  # no room -> no change
                room = self._env.layout[self._env.room_coords_to_ind(
                    room_x-1, room_y)]
                pos = self._env.pos_coords_to_ind(self._env.GRID_SIZE-1, pos_y)
                return room, pos
            pos = self._env.pos_coords_to_ind(pos_x-1, pos_y)
            return room, pos
        if action == 3:  # down
            if pos_x == self._env.GRID_SIZE-1:  # change rooms
                if room_x == self._env.SUPERGRID_SIZE-1:
                    return room, pos  # no room -> no change
                room = self._env.layout[self._env.room_coords_to_ind(
                    room_x+1, room_y)]
                pos = self._env.pos_coords_to_ind(0, pos_y)
                return room, pos
            pos = self._env.pos_coords_to_ind(pos_x+1, pos_y)
            return room, pos
        return room, pos  # slay action -> no change

    def _update_enemy(self, pos):
        pos_x, pos_y = self._env.pos_ind_to_coords(pos)
        if (pos_y % 2 == 0 and pos_x == self._env.GRID_SIZE-1) or \
           pos_y % 2 == 1 and pos_x == 0:
            if pos_y == self._env.GRID_SIZE-1:
                return self._env.pos_coords_to_ind(0, 0)
            return self._env.pos_coords_to_ind(pos_x, pos_y+1)
        if pos_y % 2 == 0:
            return self._env.pos_coords_to_ind(pos_x+1, pos_y)
        return self._env.pos_coords_to_ind(pos_x-1, pos_y)

    def get_random_constrained_transition(self, constraint):
        num_tries = 0
        while True:
            state_dict = {
                self._env.robot_room: self._env.robot_room.sample(),
                self._env.robot_pos: self._env.robot_pos.sample(),
            }
            try:
                enemy_room_ind = self._env.enemy_rooms.index(
                    state_dict[self._env.robot_room])+1
            except ValueError:
                enemy_room_ind = 0  # robot not in same room as an enemy
            state_dict[self._env.robot_enemy_room] = enemy_room_ind
            for room in self._env.enemy_rooms:
                for enemy_pos in self._env.enemy_poses[room]:
                    state_dict[enemy_pos] = enemy_pos.sample()
            state = self._env.state_factory.build(state_dict)
            if constraint.check(state):
                break
            num_tries += 1
            if num_tries > 1000:
                return None
        action = self._env.action_var.sample()
        return state, action

    def update_constrained_transition(self, state, action, var):
        if var is self._env.action_var:
            action = self._env.action_var.sample()
        elif var == self._env.robot_room:  # dependence on robot_enemy_room
            state = state.update(var, var.sample())
            try:
                enemy_room_ind = self._env.enemy_rooms.index(state[var])+1
            except ValueError:
                enemy_room_ind = 0  # robot not in same room as an enemy
            state = state.update(self._env.robot_enemy_room, enemy_room_ind)
        elif var == self._env.robot_enemy_room:  # dependence on robot_room
            state = state.update(var, var.sample())
            if state[var] > 0:
                state = state.update(
                    self._env.robot_room, self._env.enemy_rooms[state[var]-1])
        else:
            state = state.update(var, var.sample())
        return state, action


class EnemiesEnvFamily(Environment):
    """Enemies environment family definition.
    """
    SUPERGRID_SIZE = 3
    GRID_SIZE = None
    NUM_ENEMY_ROOMS = None
    ENEMIES_PER_ROOM = None
    ENEMY_MOVE_PROB = 0.9
    SUCCESS_REWARD = None
    INIT_REWARD = None
    INIT = (0, 0)  # (room, pos)
    GOAL = (2, 0)  # (room, pos)

    def __init__(self, generate_csi=True, layout_params=(None, None)):
        self.robot_room = StateVariable(
            "robotroom", self.SUPERGRID_SIZE*self.SUPERGRID_SIZE)
        self.robot_pos = StateVariable("robotpos",
                                       self.GRID_SIZE*self.GRID_SIZE)
        self.robot_enemy_room = StateVariable(
            "robotenemyroom", self.NUM_ENEMY_ROOMS+1)  # variable to constrain
        self.state_variables = [self.robot_room, self.robot_pos,
                                self.robot_enemy_room]
        self.enemy_rooms = []
        self.enemy_poses = {}
        all_rooms = [i for i in range(self.SUPERGRID_SIZE*self.SUPERGRID_SIZE)
                     if i not in (self.INIT[0], self.GOAL[0])]
        for i in range(self.NUM_ENEMY_ROOMS):
            room = all_rooms[i]
            self.enemy_rooms.append(room)
            self.enemy_poses[room] = set()
            for j in range(self.ENEMIES_PER_ROOM[i]):
                enemy_pos = StateVariable("enemy{}room{}pos".format(j, room),
                                          self.GRID_SIZE*self.GRID_SIZE)
                self.state_variables.append(enemy_pos)
                self.enemy_poses[room].add(enemy_pos)
        self.state_factory = StateFactory(self.state_variables)
        self.num_icons = 3  # agent = 0, enemy = 1, goal = 2
        self.icon_images = None
        self.icon_images1 = OrderedDict([
            (0, load_asset("keys_agent.png")),
            (1, load_asset("keys_dragon.png")),
            (2, load_asset("keys_goal.png")),
        ])
        self.icon_images2 = OrderedDict([
            (0, load_asset("keys_agent_ud.png")),
            (1, load_asset("keys_dragon.png")),
            (2, load_asset("keys_goal.png")),
        ])
        # Layout_params is a pair of ("easy"/"medium"/"hard", index).
        self._layout_params = layout_params
        self._initialize_layout()
        transition_model = EnemiesTransitionModel(self)
        reward_fn = self._construct_reward_fn()
        super().__init__(transition_model, reward_fn, self.state_factory,
                         generate_csi=generate_csi)

    def sample_initial_state(self):
        return self.initial_state

    def get_solver_info(self, relaxation=None):
        if sc.solver_name == "AsyncValueIteration" and relaxation is not None:
            return {"imposed_constraint": relaxation[0].imposed_constraint}
        return None

    def render(self, state):
        grid = np.zeros((self.SUPERGRID_SIZE*self.GRID_SIZE,
                         self.SUPERGRID_SIZE*self.GRID_SIZE,
                         self.num_icons), dtype=bool)
        room_x, room_y = self.room_ind_to_coords(
            self.rev_layout[state[self.robot_room]])
        pos_x, pos_y = self.pos_ind_to_coords(state[self.robot_pos])
        grid[room_x*self.GRID_SIZE+pos_x,
             room_y*self.GRID_SIZE+pos_y, 0] = 1
        for room in self.enemy_rooms:
            room_x, room_y = self.room_ind_to_coords(self.rev_layout[room])
            for enemy_pos in self.enemy_poses[room]:
                pos_x, pos_y = self.pos_ind_to_coords(state[enemy_pos])
                grid[room_x*self.GRID_SIZE+pos_x,
                     room_y*self.GRID_SIZE+pos_y, 1] = 1
        room_x, room_y = self.room_ind_to_coords(self.rev_layout[self.GOAL[0]])
        pos_x, pos_y = self.pos_ind_to_coords(self.GOAL[1])
        grid[room_x*self.GRID_SIZE+pos_x,
             room_y*self.GRID_SIZE+pos_y, 2] = 1
        # Render agent upside-down if in a room with enemies.
        if state[self.robot_enemy_room] == 0:
            self.icon_images = self.icon_images1
        else:
            self.icon_images = self.icon_images2
        fig = render_from_layout(grid, self._get_token_images,
                                 background_grid=False,
                                 line_every=self.GRID_SIZE)
        return fig

    def heuristic(self, state):
        return 0

    def room_coords_to_ind(self, room_x, room_y):
        """Room (x, y) to a single index.
        """
        return room_x*self.SUPERGRID_SIZE+room_y

    def room_ind_to_coords(self, room_ind):
        """Room single index to (x, y).
        """
        return room_ind//self.SUPERGRID_SIZE, room_ind%self.SUPERGRID_SIZE

    def pos_coords_to_ind(self, pos_x, pos_y):
        """Pos (x, y) to a single index.
        """
        return pos_x*self.GRID_SIZE+pos_y

    def pos_ind_to_coords(self, pos_ind):
        """Pos single index to (x, y).
        """
        return pos_ind//self.GRID_SIZE, pos_ind%self.GRID_SIZE

    def _get_token_images(self, obs_cell):
        for cell_type, img in self.icon_images.items():
            if obs_cell[cell_type]:
                yield img

    def _construct_reward_fn(self):
        reward_vars = [self.robot_room, self.robot_pos]
        features = self._get_theta()
        def reward_fn(state, action):
            _ = action  # unused
            if state[self.robot_room] == self.GOAL[0] and \
                 state[self.robot_pos] == self.GOAL[1]:
                reward = self.SUCCESS_REWARD
            elif state[self.robot_room] == self.INIT[0] and \
                 state[self.robot_pos] == self.INIT[1]:
                reward = self.INIT_REWARD
            else:
                reward = 0.0
            done = reward > 0.0
            return reward, done
        return RewardFunction(reward_vars, reward_fn, features)

    def _initialize_layout(self):
        global EASY, MEDIUM, HARD
        rand_state = np.random.RandomState(seed=0)
        if EASY is None:
            # Generate all possible layouts.
            perms = itertools.permutations(range(self.SUPERGRID_SIZE*
                                                 self.SUPERGRID_SIZE))
            def _filter_fn(perm):
                # Always keep init and goal supersquares in same place.
                return (perm[self.INIT[0]] == self.INIT[0] and
                        perm[self.GOAL[0]] == self.GOAL[0])
            perms = list(filter(_filter_fn, perms))
            rand_state.shuffle(perms)
            # Organize into:
            # 1) direct path is free (easy);
            # 2) some non-direct path is free (medium).
            # 3) all paths are blocked (hard);
            EASY, MEDIUM, HARD = [], [], []
            for perm in perms:
                if self._direct_path_free(perm):
                    EASY.append(perm)
                elif self._all_paths_blocked(perm):
                    HARD.append(perm)
                else:
                    MEDIUM.append(perm)
            del perms
        # Select one.
        if self._layout_params[0] == "easy":
            perm = EASY[self._layout_params[1]]
        elif self._layout_params[0] == "medium":
            perm = MEDIUM[self._layout_params[1]]
        elif self._layout_params[0] == "hard":
            perm = HARD[self._layout_params[1]]
        else:
            raise Exception("Unexpected layout_params: {}".format(
                self._layout_params))
        self.layout = dict(enumerate(perm))
        self.rev_layout = {v: k for k, v in self.layout.items()}
        init_state_dict = {
            self.robot_room: self.INIT[0],
            self.robot_pos: self.INIT[1],
            self.robot_enemy_room: 0,
        }
        for room in self.enemy_rooms:
            for enemy_pos in self.enemy_poses[room]:
                init_state_dict[enemy_pos] = rand_state.choice(enemy_pos.size)
        self.initial_state = self.state_factory.build(init_state_dict)

    def _direct_path_free(self, perm):
        layout = dict(enumerate(perm))
        cur_x, cur_y = self.room_ind_to_coords(self.INIT[0])
        goal_x, goal_y = self.room_ind_to_coords(self.GOAL[0])
        while cur_x != goal_x or cur_y != goal_y:
            # First traverse x, then traverse y.
            if cur_x < goal_x:
                cur_x += 1
            elif cur_x > goal_x:
                cur_x -= 1
            elif cur_y < goal_y:
                cur_y += 1
            elif cur_y > goal_y:
                cur_y -= 1
            else:
                raise Exception("Can't get here")
            room_ind = self.room_coords_to_ind(cur_x, cur_y)
            if layout[room_ind] in self.enemy_rooms:
                return False
        return True

    def _all_paths_blocked(self, perm):
        layout = dict(enumerate(perm))
        # Run BFS in the supergrid to find an unblocked path.
        queue = [self.INIT[0]]
        visited = set()
        while queue:
            room_ind = queue.pop(0)
            visited.add(room_ind)
            cur_x, cur_y = self.room_ind_to_coords(room_ind)
            children = []
            if cur_x > 0:
                children.append((cur_x-1, cur_y))
            if cur_x < self.SUPERGRID_SIZE-1:
                children.append((cur_x+1, cur_y))
            if cur_y > 0:
                children.append((cur_x, cur_y-1))
            if cur_y < self.SUPERGRID_SIZE-1:
                children.append((cur_x, cur_y+1))
            for next_x, next_y in children:
                next_room_ind = self.room_coords_to_ind(next_x, next_y)
                if next_room_ind in visited:
                    continue
                if layout[next_room_ind] in self.enemy_rooms:
                    continue
                if next_room_ind == self.GOAL[0]:
                    return False
                queue.append(next_room_ind)
        return True

    def _get_theta(self):
        # Channels: 0 = init, 1-N = enemy rooms, -1 = goal.
        # Note that channel dimension is FIRST due to pytorch rules.
        supergrid = np.zeros((1+len(self.enemy_rooms)+1,
                              self.SUPERGRID_SIZE,
                              self.SUPERGRID_SIZE))
        start_x, start_y = self.room_ind_to_coords(self.INIT[0])
        supergrid[0, start_x, start_y] = 1
        goal_x, goal_y = self.room_ind_to_coords(self.GOAL[0])
        supergrid[-1, goal_x, goal_y] = 1
        for i, room in enumerate(self.enemy_rooms):
            room_x, room_y = self.room_ind_to_coords(self.rev_layout[room])
            supergrid[i+1, room_x, room_y] = 1
        return supergrid


class EnemiesEnvFamilyBig(EnemiesEnvFamily):
    """Big version.
    """
    GRID_SIZE = 3
    NUM_ENEMY_ROOMS = 2
    SUCCESS_REWARD = 100.0
    INIT_REWARD = -5.0
    ENEMIES_PER_ROOM = [1, 2]


class EnemiesEnvFamilySmall(EnemiesEnvFamily):
    """Small version.
    """
    GRID_SIZE = 2
    NUM_ENEMY_ROOMS = 2
    SUCCESS_REWARD = 50.0
    INIT_REWARD = -20.0
    ENEMIES_PER_ROOM = [2, 3]


def create_enemies_env(base_class, layout_params, global_context):
    """Method for dynamically creating an EnemiesEnv class.
    """
    name = "{}{}{}".format(base_class.__name__,
                           layout_params[0].capitalize(),
                           layout_params[1])
    def __init__(self, generate_csi=True):
        base_class.__init__(self, generate_csi=generate_csi,
                            layout_params=layout_params)
    newclass = type(name, (base_class,), {"__init__" : __init__})
    global_context[newclass.__name__] = newclass
