"""Implementation of task and motion planning (TAMP) NAMO
family of environments.
"""

import time
import glob
import os
import numpy as np
import pybullet as p
from settings import GeneralConfig as gc
from envs.env_base import Environment, TransitionModel, RewardFunction
from envs.assets import p_constants
from structs import StateFactory, DiscreteVariable, ContinuousVariable, \
    MultiDimVariable, StateVariable, ContinuousStateVariable
from solvers.rrt import BiRRT
from utils import lin_interp

SCALE = 3
DOOR_SCALE = 0.25  # fraction between 0 and 1
OBJ_SIZE = 0.3
OBJ_HEIGHT = 0.7
INCLUSION_PROB = 0.5  # prob of each object being active
CLEAR_BP_DIST = 0.125  # how far to be from object when clearing it
ROBOT_Z = 0.3
NUM_SYMBOLIC_POSES = 15
IMG_DIM = 50


class TAMPNAMOTransitionModel(TransitionModel):
    """Transition model for the TAMP NAMO environments.
    """
    def __init__(self, env):
        self._env = env
        self._constrained_env = None  # used for CSI learning
        self._p_objs = None
        self._p_held_obj_tf = None
        self._rrt = None
        self._initialize_pybullet()
        self._allowed_rooms = None
        self._rrt_cache = {}

    def get_state_vars(self):
        return self._env.state_variables

    def get_action_var(self, state_vars):
        return self._env.ACT

    def model(self, state, action):
        next_state = self.sample_next_state(state, action)
        yield next_state, 1.0

    def ml_next_state(self, state, action):
        raise Exception("BFSReplan incompatible with a TAMP environment")

    def sample_next_state(self, state, action):
        next_state, _ = self.sample_next_state_failure(state, action)
        return next_state

    def sample_next_state_failure(self, state, action):
        """Returns a tuple of (next_state, reason for failure).
        Possible failures:
        - None, meaning no failure
        - ("wallcollision", <target object name>)
        - ("objcollision", <target object name>, <first collision encountered>)
        """
        # Forcibly set pybullet state.
        self.force_pybullet_state(state)
        assert len(action) == 3
        obj_ind, clear_base_pose, target_base_pose = action
        self._allowed_rooms = set()
        for room in range(9):
            if all(max(state[obj]) > -50 for obj, _
                   in self._iterate_objs_in_room(room)):
                # If we haven't ignored all the objects in this room,
                # allow ourselves to go in it.
                self._allowed_rooms.add(room)
        # Take action in pybullet.
        failure = self._update_pybullet_state(obj_ind, clear_base_pose,
                                              target_base_pose)
        next_state_dict = state.todict()
        if failure is not None:
            next_state = state.state_factory.build(next_state_dict)
            return next_state, failure  # failed; return failure info
        # Extract next-state information from pybullet.
        (world_x, world_y, z), orn = p.getBasePositionAndOrientation(
            self._p_robot)
        room, x, y = self._world_to_state_frame(world_x, world_y)
        next_state_dict[self._env.r_base] = np.r_[x, y, z, orn]
        next_state_dict[self._env.r_room] = room
        cur_room = 0
        counter = 0
        for obj, p_obj in zip(self._env.objs, self._p_objs):
            (world_x, world_y, z), orn = p.getBasePositionAndOrientation(p_obj)
            room, x, y = self._world_to_state_frame(world_x, world_y)
            # assert room == cur_room, "Object room cannot change"
            next_state_dict[obj] = np.r_[x, y, z, orn]
            counter += 1
            if counter == self._env.num_obj_per_room:
                cur_room += 1
                counter = 0
        next_state = state.state_factory.build(next_state_dict)
        return next_state, None

    def get_random_constrained_transition(self, constraint):
        assert INCLUSION_PROB == 1.0
        self._constrained_env = TAMPNAMOProblemArbitrary(generate_csi=False)
        state = self._constrained_env.sample_initial_state()
        num_tries = 0
        while True:
            action = self._constrained_env.action_var.sample()
            next_state, failure = self.sample_next_state_failure(state, action)
            if failure is not None:
                continue
            if constraint.check(next_state):
                break
            num_tries += 1
            if num_tries > 1000:
                return None
        return state, action

    def update_constrained_transition(self, state, action, var):
        if var is self._env.action_var:
            action = self._constrained_env.action_var.sample()
        else:
            other_state = TAMPNAMOProblemArbitrary(
                generate_csi=False).sample_initial_state()
            state = state.update(var, other_state[var])
        return state, action

    def _initialize_pybullet(self):
        if p.getConnectionInfo()["isConnected"]:
            self._p_robot, self._p_walls, self._p_objs = p_constants.PS_LIST
            self._setup_motion_planner()
            return
        if gc.do_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath("envs/assets/")
        p.loadURDF("plane.urdf")
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        self._p_robot, = p.loadSDF("kuka_with_gripper_recolored.sdf")
        p.resetBasePositionAndOrientation(self._p_robot,
                                          [0, 0, ROBOT_Z],
                                          [0, 0, 0, 1])
        # Make URDF files.
        wall_height = 3
        with open("envs/assets/side_wall_horiz.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                SCALE*3, 0.01, wall_height, 0.5, 0.5, 0.5, 1))
        with open("envs/assets/side_wall_vert.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                0.01, SCALE*3, wall_height, 0.5, 0.5, 0.5, 1))
        with open("envs/assets/inner_wall_horiz.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                SCALE*(1-DOOR_SCALE)/2, 0.01, wall_height, 0.5, 0.5, 0.5, 1))
        with open("envs/assets/inner_wall_vert.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                0.01, SCALE*(1-DOOR_SCALE)/2, wall_height, 0.5, 0.5, 0.5, 1))
        with open("envs/assets/blue_object.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                OBJ_SIZE, OBJ_SIZE, OBJ_HEIGHT, 0, 0, 1, 1))
        with open("envs/assets/red_object.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                OBJ_SIZE, OBJ_SIZE, OBJ_HEIGHT, 1, 0, 0, 1))
        # Set up walls.
        self._p_walls = []
        self._p_walls.append(p.loadURDF(
            "side_wall_vert.urdf",
            np.r_[self._state_to_world_frame(1, 0, 0.5), 0]))
        self._p_walls.append(p.loadURDF(
            "side_wall_vert.urdf",
            np.r_[self._state_to_world_frame(7, 1, 0.5), 0]))
        self._p_walls.append(p.loadURDF(
            "side_wall_horiz.urdf",
            np.r_[self._state_to_world_frame(3, 0.5, 0), 0]))
        self._p_walls.append(p.loadURDF(
            "side_wall_horiz.urdf",
            np.r_[self._state_to_world_frame(5, 0.5, 1), 0]))
        for room in [1, 2, 4, 5, 7, 8]:
            for x in [(1-DOOR_SCALE)/4, 1-(1-DOOR_SCALE)/4]:
                self._p_walls.append(p.loadURDF(
                    "inner_wall_horiz.urdf",
                    np.r_[self._state_to_world_frame(room, x, 0), 0]))
        for room in [3, 4, 5, 6, 7, 8]:
            for y in [(1-DOOR_SCALE)/4, 1-(1-DOOR_SCALE)/4]:
                self._p_walls.append(p.loadURDF(
                    "inner_wall_vert.urdf",
                    np.r_[self._state_to_world_frame(room, 0, y), 0]))
        # Set up objects.
        self._p_objs = []
        for _ in range(self._env.num_objects-1):
            self._p_objs.append(p.loadURDF("blue_object.urdf"))
        self._p_objs.append(p.loadURDF("red_object.urdf"))  # final room's obj
        for fil in glob.glob("envs/assets/*wall*.urdf"):
            os.remove(fil)
        for fil in glob.glob("envs/assets/*object*.urdf"):
            os.remove(fil)
        p_constants.set_ps([self._p_robot, self._p_walls, self._p_objs])
        self._setup_motion_planner()

    def _get_path(self, goalx, goaly):
        """Get a smoothed path to the given (goalx, goaly), obtained using the
        motion planner. Returns a tuple of (success, <path or collision set>).
        """
        cur_xy = p.getBasePositionAndOrientation(self._p_robot)[0][:2]
        precision = 4
        key = (round(cur_xy[0], precision), round(cur_xy[1], precision),
               round(goalx, precision), round(goaly, precision))
        if key in self._rrt_cache:
            return True, self._rrt_cache[key]
        path = self._rrt.query(cur_xy, [goalx, goaly], timeout=30)
        if path is None:
            # No path exists, return collision information.
            _, cols = self._rrt.query_ignore_collisions(cur_xy, [goalx, goaly])
            return False, cols
        path = self._rrt.smooth_path(path)
        self._rrt_cache[key] = path
        return True, path  # successfully found a path

    def _follow(self, path, target_orn, do_interpolate):
        """Follow the given path, yielded by self._get_path.
        Uses self._move_base().
        """
        (_, _, cur_z), cur_orn = p.getBasePositionAndOrientation(self._p_robot)
        cur_orn = p.getEulerFromQuaternion(cur_orn)[2]
        for i, loc in enumerate(path):
            interp_orn = cur_orn*(1-i/len(path))+target_orn*i/len(path)
            self._move_base(np.r_[loc, cur_z], interp_orn, do_interpolate)

    def _setup_motion_planner(self):
        def _sample_fn(_):
            entries = []
            for room in self._allowed_rooms:
                entries.append(self._state_to_world_frame(room, 0.2, 0.5))
                entries.append(self._state_to_world_frame(room, 0.8, 0.5))
                entries.append(self._state_to_world_frame(room, 0.5, 0.2))
                entries.append(self._state_to_world_frame(room, 0.5, 0.8))
            return entries[gc.rand_state.randint(len(entries))]
        def _extend_fn(pt1, pt2):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            num = int(np.ceil(max(abs(pt1-pt2))))*10
            if num == 0:
                yield pt2, True
            for i in range(1, num+1):
                yield np.r_[pt1*(1-i/num)+pt2*i/num], True
        def _collision_fn(pt):
            # NOTE: we only check wall collisions here. Purposely not
            # checking object collisions because that happens elsewhere.
            world_x, world_y = pt
            room, x, y = self._world_to_state_frame(world_x, world_y)
            if room not in self._allowed_rooms:
                return "wall"
            if (x < 0.2 and not (1-DOOR_SCALE)/2 < y < 1-(1-DOOR_SCALE)/2):
                return "wall"
            if (x > 0.8 and not (1-DOOR_SCALE)/2 < y < 1-(1-DOOR_SCALE)/2):
                return "wall"
            if (y < 0.2 and not (1-DOOR_SCALE)/2 < x < 1-(1-DOOR_SCALE)/2):
                return "wall"
            if (y > 0.8 and not (1-DOOR_SCALE)/2 < x < 1-(1-DOOR_SCALE)/2):
                return "wall"
            return None
        def _distance_fn(pt1, pt2):
            return (pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2
        self._rrt = BiRRT(_sample_fn, _extend_fn, _collision_fn, _distance_fn)

    def force_pybullet_state(self, state):
        """Forcibly set pybullet state to the given one.
        """
        # Set robot pose.
        r_base = state[self._env.r_base]
        r_room = state[self._env.r_room]
        world_x, world_y = self._state_to_world_frame(
            r_room, r_base[0], r_base[1])
        cur_z = p.getBasePositionAndOrientation(self._p_robot)[0][2]
        target_orn = p.getEulerFromQuaternion(r_base[3:])[2]
        self._move_base([world_x, world_y, cur_z], target_orn,
                        do_interpolate=False)
        # Set object poses.
        cur_room = 0
        counter = 0
        for obj, p_obj in zip(self._env.objs, self._p_objs):
            pose = state[obj]
            world_x, world_y = self._state_to_world_frame(
                cur_room, pose[0], pose[1])
            p.resetBasePositionAndOrientation(
                p_obj, np.r_[world_x, world_y, pose[2]], pose[3:])
            counter += 1
            if counter == self._env.num_obj_per_room:
                cur_room += 1
                counter = 0

    def _update_pybullet_state(self, obj_ind, clear_base_pose,
                               target_base_pose):
        """Returns None if updating state was successful, and otherwise returns
        the reason for failure.
        """
        if obj_ind >= len(self._env.objs) or obj_ind < 0:
            return None
        target_obj_name = self._env.objs[obj_ind].name
        room = obj_ind // self._env.num_obj_per_room
        do_interpolate = (p.getConnectionInfo()["connectionMethod"] == p.GUI and
                          gc.do_render)
        # Move to clear_base_pose.
        world_x, world_y = self._state_to_world_frame(room, clear_base_pose[0],
                                                      clear_base_pose[1])
        succ, path = self._get_path(world_x, world_y)
        if not succ:
            return ("wallcollision", target_obj_name)
        # Check path for object collisions before actually moving.
        for pt in path:
            room, _, _ = self._world_to_state_frame(pt[0], pt[1])
            for obj, p_obj in self._iterate_objs_in_room(room):
                if self._is_in_collision(pt, p_obj):
                    return ("objcollision", target_obj_name, obj.name)
        self._follow(path, p.getEulerFromQuaternion(clear_base_pose[3:])[2],
                     do_interpolate)
        # Move object out of the way.
        base_pos, base_orn = p.getBasePositionAndOrientation(self._p_robot)
        target_loc, target_orn = p.multiplyTransforms(
            base_pos, base_orn, [0, CLEAR_BP_DIST*3, 0.65], [0, 1, 0, 0])
        self._move_end_effector(target_loc, target_orn, do_interpolate)
        self._grab_object(self._p_objs[obj_ind])
        succ, path = self._get_path(base_pos[0], base_pos[1])
        assert succ, "Spinning in place should always succeed"
        self._follow(path, p.getEulerFromQuaternion(base_orn)[2]+2*np.pi,
                     do_interpolate)  # spin in place
        self._grab_object(None)
        self._reset_joint_state(num_steps=(2500 if do_interpolate else 1))
        # Move to target_base_pose.
        world_x, world_y = self._state_to_world_frame(room, target_base_pose[0],
                                                      target_base_pose[1])
        succ, path = self._get_path(world_x, world_y)
        if not succ:
            return ("wallcollision", target_obj_name)
        self._follow(path, p.getEulerFromQuaternion(target_base_pose[3:])[2],
                     do_interpolate)
        return None

    def _iterate_objs_in_room(self, room):
        num = self._env.num_obj_per_room
        return zip(self._env.objs[room*num:(room+1)*num],
                   self._p_objs[room*num:(room+1)*num])

    @staticmethod
    def _is_in_collision(pt, p_obj):
        (obj_x, obj_y, obj_z), _ = p.getBasePositionAndOrientation(p_obj)
        if obj_z < 0:
            return False
        # NOTE: using full extents instead of half extents intentionally,
        # so robot can't get too close to an object.
        return abs(obj_x-pt[0]) < OBJ_SIZE and abs(obj_y-pt[1]) < OBJ_SIZE

    @staticmethod
    def _state_to_world_frame(room, x, y):
        room_x, room_y = room//3, room%3
        return [SCALE*(room_x+x)-SCALE*1.5, SCALE*(room_y+y)-SCALE*1.5]

    @staticmethod
    def _world_to_state_frame(world_x, world_y):
        room_x_plus_x = (world_x+SCALE*1.5)/SCALE
        room_y_plus_y = (world_y+SCALE*1.5)/SCALE
        room_x, x = int(room_x_plus_x), room_x_plus_x-int(room_x_plus_x)
        room_y, y = int(room_y_plus_y), room_y_plus_y-int(room_y_plus_y)
        room = 3*room_x+room_y
        return room, x, y

    def _move_base(self, target_loc, target_orn, do_interpolate):
        orig_loc, orig_orn = p.getBasePositionAndOrientation(self._p_robot)
        orig_orn = p.getEulerFromQuaternion(orig_orn)[2]
        orig = np.r_[orig_loc, orig_orn]
        target = np.r_[target_loc, target_orn]
        dist = np.linalg.norm(target-orig)
        if do_interpolate:
            speed = 0.05
            num_steps = int(np.ceil(abs(dist)/speed))
        else:
            num_steps = 1
        for i in range(1, num_steps+1):
            interpolated = orig*(1-i/num_steps)+target*i/num_steps
            interp_orien = p.getQuaternionFromEuler([0, 0, interpolated[3]])
            p.resetBasePositionAndOrientation(
                self._p_robot, interpolated[:3], interp_orien)
            base_link = np.r_[p.getLinkState(self._p_robot, 9)[:2]]
            if self._p_held_obj_tf is not None:
                obj_id, transf = self._p_held_obj_tf
                obj_loc, obj_orn = p.multiplyTransforms(
                    base_link[:3], base_link[3:], transf[0], transf[1])
                p.resetBasePositionAndOrientation(
                    obj_id, obj_loc, obj_orn)
            if do_interpolate:
                time.sleep(0.01)

    def _reset_joint_state(self, num_steps):
        njo = min(p.getNumJoints(self._p_robot), 7)
        old_joints = np.array([p.getJointState(self._p_robot, joint_idx)[0]
                               for joint_idx in range(njo)])
        new_joints = np.zeros(njo)
        for i in range(1, num_steps+1):
            interpolated = old_joints*(1-i/num_steps)+new_joints*i/num_steps
            for joint_idx in range(njo):
                p.resetJointState(self._p_robot, joint_idx,
                                  interpolated[joint_idx])

    def _move_end_effector(self, target_loc, target_orn, do_interpolate,
                           open_amt=None, avoid_cols=False, succ_thresh=0.05):
        num_iter = 0
        while True:
            num_iter += 1
            if num_iter > 10000:
                return False
            njo = min(p.getNumJoints(self._p_robot), 7)
            old_joints = np.array([p.getJointState(self._p_robot, joint_idx)[0]
                                   for joint_idx in range(njo)])
            new_joints = np.array(p.calculateInverseKinematics(
                self._p_robot, endEffectorLinkIndex=6,
                targetPosition=target_loc,
                targetOrientation=target_orn, solver=0))[:njo]
            old_open_amt = p.getJointState(self._p_robot, 11)[0]
            dist = np.linalg.norm(new_joints-old_joints)
            if open_amt is not None:
                dist = max(dist, abs(old_open_amt-open_amt))
            dist_thresh = 1e-2
            if avoid_cols:
                assert open_amt is None
                dist_thresh = 1e-1
            if dist < dist_thresh:
                link_pos, link_orn = p.getLinkState(self._p_robot, 6)[-2:]
                error = np.linalg.norm(np.r_[link_pos, link_orn]-
                                       np.r_[target_loc, target_orn])
                if error < succ_thresh:
                    return True  # success: reached target pose
                return False  # failure: didn't reach target pose
            if do_interpolate:
                speed = 0.05
                num_steps = int(np.ceil(abs(dist)/speed))
            else:
                num_steps = 1
            for i in range(1, num_steps+1):
                interpolated = old_joints*(1-i/num_steps)+new_joints*i/num_steps
                if avoid_cols:
                    p.setJointMotorControlArray(
                        self._p_robot,
                        range(njo),
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=new_joints,
                        targetVelocities=[0 for _ in range(njo)],
                        forces=[500 for _ in range(njo)],
                        positionGains=[0.03 for _ in range(njo)],
                        velocityGains=[1 for _ in range(njo)])
                    p.stepSimulation()
                else:
                    for joint_idx in range(njo):
                        p.resetJointState(self._p_robot, joint_idx,
                                          interpolated[joint_idx])
                if open_amt is not None:
                    # Open gripper to open_amt.
                    open_interp = (old_open_amt*(1-i/num_steps)+
                                   open_amt*i/num_steps)
                    p.resetJointState(self._p_robot, 8, -open_interp)
                    p.resetJointState(self._p_robot, 11, open_interp)
                else:
                    p.resetJointState(self._p_robot, 8, -old_open_amt)
                    p.resetJointState(self._p_robot, 11, old_open_amt)
                base_link = np.r_[p.getLinkState(self._p_robot, 9)[:2]]
                if self._p_held_obj_tf is not None:
                    obj_id, transf = self._p_held_obj_tf
                    obj_loc, obj_orn = p.multiplyTransforms(
                        base_link[:3], base_link[3:], transf[0], transf[1])
                    p.resetBasePositionAndOrientation(
                        obj_id, obj_loc, obj_orn)
                if do_interpolate:
                    time.sleep(0.01)

    def _grab_object(self, obj_id):
        if obj_id is None:
            self._p_held_obj_tf = None
        else:
            base_link_to_world = np.r_[p.invertTransform(
                *p.getLinkState(self._p_robot, 9)[:2])]
            world_to_obj = np.r_[p.getBasePositionAndOrientation(obj_id)]
            self._p_held_obj_tf = (obj_id, p.multiplyTransforms(
                base_link_to_world[:3], base_link_to_world[3:],
                world_to_obj[:3], world_to_obj[3:]))

    @staticmethod
    def _pause_pybullet(secs=float("inf")):
        start_time = time.time()
        while True:
            p.setGravity(0, 0, -10)
            time.sleep(0.1)
            if time.time()-start_time > secs:
                break


class TAMPNAMOEnvFamily(Environment):
    """TAMP NAMO environment family definition.
    """
    # We'll always have 9 rooms arranged into a 3x3 grid.
    # Rooms are numbered 0-8.
    # Generate 5 possible objects per room, except the last room always
    # has 1 object -> 41 objects total, numbered 0-40.
    # Objects 0-4 are room 0's, ..., 34-39 are room 7's, 40 is room 8's.
    # Initial state will decide how many objects are actually in each room.
    # Goal is to get to the location of object 40.
    num_obj_per_room = 5
    num_objects = num_obj_per_room*8+1
    se3_bounds = [np.ones(7)*-100, np.ones(7)*100]
    # Action: object to clear, base pose for clearing, target base pose.
    ACT_OBJ = DiscreteVariable("act_obj", num_objects)
    ACT_CLEAR = ContinuousVariable("act_clear", se3_bounds)
    ACT_TARGET = ContinuousVariable("act_target", se3_bounds)
    state_variables = []
    # Robot base pose.
    r_base = ContinuousStateVariable("robotbase", se3_bounds)
    state_variables.append(r_base)
    # Robot room.
    r_room = StateVariable("robotroom", 9)
    state_variables.append(r_room)
    # Objects.
    objs = []
    for i in range(num_objects):
        obj = ContinuousStateVariable("obj{}".format(i), se3_bounds)
        objs.append(obj)
        state_variables.append(obj)
    state_factory = StateFactory(state_variables)

    def __init__(self, generate_csi=True, myseed=None):
        self._myseed = myseed
        (self._initial_state,
         self._goal,
         self._goal_pddl) = self._construct_initstate_and_goal()
        reward_fn = self._construct_reward_fn()
        self.reward = reward_fn
        transition_model = TAMPNAMOTransitionModel(self)
        transition_model.force_pybullet_state(self._initial_state)
        act_domain = []
        # NOTE: should match below (1)
        for i, obj in enumerate(self.objs):
            pose = self._initial_state[obj]
            if pose[2] < 0:  # object not included in room
                continue
            quat = p.getQuaternionFromEuler([0, 0, -np.pi/2])
            act_domain.append(
                [i, np.r_[pose[0]-CLEAR_BP_DIST, pose[1], ROBOT_Z, quat],
                 np.r_[pose[0], pose[1], 0, quat]])
            quat = p.getQuaternionFromEuler([0, 0, np.pi/2])
            act_domain.append(
                [i, np.r_[pose[0]+CLEAR_BP_DIST, pose[1], ROBOT_Z, quat],
                 np.r_[pose[0], pose[1], 0, quat]])
            quat = p.getQuaternionFromEuler([0, 0, 0])
            act_domain.append(
                [i, np.r_[pose[0], pose[1]-CLEAR_BP_DIST, ROBOT_Z, quat],
                 np.r_[pose[0], pose[1], 0, quat]])
            quat = p.getQuaternionFromEuler([0, 0, np.pi])
            act_domain.append(
                [i, np.r_[pose[0], pose[1]+CLEAR_BP_DIST, ROBOT_Z, quat],
                 np.r_[pose[0], pose[1], 0, quat]])
        self.ACT = MultiDimVariable(
            "action", [self.ACT_OBJ, self.ACT_CLEAR, self.ACT_TARGET],
            act_domain)
        super().__init__(transition_model, reward_fn, self.state_factory,
                         generate_csi=generate_csi)

    def get_solver_info(self, relaxation=None):
        """This method gives special methods needed by the TAMP solver.
        """
        info = {}

        def _get_domain_pddl():
            return """
            (define (domain tampnamodomain)
              (:requirements :strips :typing)
              (:types object pose)
              (:predicates
                (robotat ?p - pose)
                (at ?o - object ?p - pose)
                (ispose ?o - object ?cbp - pose ?tbp - pose)
                (obstructs ?o - object ?p - pose)
              )
              (:action moveclear
                :parameters (?o - object ?rp - pose ?cbp - pose ?tbp - pose)
                :precondition (and (robotat ?rp)
                                   (at ?o ?tbp)
                                   (ispose ?o ?cbp ?tbp)
                                   (forall (?o2 - object) (not (obstructs ?o2 ?cbp)))
                                   (forall (?o2 - object) (not (obstructs ?o2 ?tbp)))
                              )
                :effect (and (not (robotat ?rp))
                             (robotat ?tbp)
                             ; (not (at ?o ?tbp))
                             ; (at ?o cbp)
                             (forall (?p2 - pose) (not (obstructs ?o ?p2)))
                        )
              )
            )"""
        info["get_domain_pddl"] = _get_domain_pddl

        def _get_problem_pddl(state, discovered_facts):
            state = state.todict()
            objects = ["robotinitpose - pose"]
            state_pddl = ["(robotat robotinitpose)"]
            for key in state:
                if key in self.objs:
                    name = key.name
                    objects.append("{} - object".format(name))
                    for i in range(NUM_SYMBOLIC_POSES):
                        objects.append("clearpose_{}_{} - pose".format(i, name))
                        state_pddl.append("(ispose {1} clearpose_{0}_{1} "
                                          "startpose_{1})".format(i, name))
                    objects.append("startpose_{} - pose".format(name))
                    state_pddl.append("(at {0} startpose_{0})".format(name))
            state_pddl.extend(discovered_facts)
            objects = "\n\t".join(objects)
            state_pddl = "\n\t".join(state_pddl)
            return """
            (define (problem tampnamoproblem)
              (:domain tampnamodomain)
              (:objects\n\t{}
              )
              (:goal {})
              (:init\n\t{}
              )
            )""".format(objects, self._goal_pddl, state_pddl)
        info["get_problem_pddl"] = _get_problem_pddl

        def _refine_step(state, symbolic_action):
            act = symbolic_action.split()
            obj_ind = [i for i, o in enumerate(self.objs)
                       if o.name == act[1]][0]
            # Generate candidate base poses for clearing.
            # NOTE: should match above (1)
            pose = state[self.objs[obj_ind]]
            assert pose[2] > 0  # object should always be included in room
            poses = []
            quat = p.getQuaternionFromEuler([0, 0, -np.pi/2])
            poses.append(np.r_[pose[0]-CLEAR_BP_DIST, pose[1], ROBOT_Z, quat])
            quat = p.getQuaternionFromEuler([0, 0, np.pi/2])
            poses.append(np.r_[pose[0]+CLEAR_BP_DIST, pose[1], ROBOT_Z, quat])
            quat = p.getQuaternionFromEuler([0, 0, 0])
            poses.append(np.r_[pose[0], pose[1]-CLEAR_BP_DIST, ROBOT_Z, quat])
            quat = p.getQuaternionFromEuler([0, 0, np.pi])
            poses.append(np.r_[pose[0], pose[1]+CLEAR_BP_DIST, ROBOT_Z, quat])
            # Select a random base pose.
            clear_pose = poses[gc.rand_state.randint(len(poses))]
            target_pose = np.r_[pose[0], pose[1], ROBOT_Z, clear_pose[3:]]
            action = [obj_ind, clear_pose, target_pose]
            return action
        info["refine_step"] = _refine_step

        def _failure_to_facts(failure):
            if failure[0] == "wallcollision":
                return None  # no fact to discover, just a bad refinement
            if failure[0] == "objcollision":
                target, col = failure[1], failure[2]
                if target == col:
                    return None  # no fact to discover, just a bad refinement
                facts = []
                for i in range(NUM_SYMBOLIC_POSES):
                    facts.append("(obstructs {} clearpose_{}_{})".format(
                        col, i, target))
                facts.append("(obstructs {} startpose_{})".format(col, target))
                return facts
            raise Exception("Unexpected failure: {}".format(failure))
        info["failure_to_facts"] = _failure_to_facts

        return info

    def sample_initial_state(self):
        self.transition_model.force_pybullet_state(self._initial_state)
        return self._initial_state

    def render(self, state):
        pass

    def _construct_reward_fn(self):
        reward_vars = [self.r_base, self.r_room, self.objs[-1]]
        _, features = self._get_reward_features()
        def reward_fn(state, action):
            _ = action  # unused
            reward = 0.0
            if self._goal(state):
                reward = 1000.0
            done = reward != 0
            return reward, done
        return RewardFunction(reward_vars, reward_fn, features)

    def _construct_initstate_and_goal(self):
        state_dict = {}
        state_dict[self.r_base] = [0.5, 0.5, ROBOT_Z, 0, 0, 0, 1]
        state_dict[self.r_room] = 0  # start in first room
        obj_pose_data, _ = self._get_reward_features()
        for i, obj in enumerate(self.objs):
            obj_x, obj_y, obj_z = obj_pose_data[3*i:3*(i+1)]
            state_dict[obj] = [obj_x, obj_y, obj_z, 0, 0, 0, 1]
        init_state = self.state_factory.build(state_dict)
        goal_x, goal_y = init_state[self.objs[-1]][:2]  # room 8 object's loc
        def _goal(state):
            robot_x, robot_y = state[self.r_base][:2]
            robot_room = state[self.r_room]
            return (robot_room == 8 and
                    (robot_x-goal_x)**2+(robot_y-goal_y)**2 < 0.001)
        goal_pddl = "(robotat startpose_{})".format(self.objs[-1])
        return init_state, _goal, goal_pddl

    def _get_reward_features(self):
        rand_state = np.random.RandomState(seed=self._myseed)
        obj_pose_data = []
        # Image has only one channel, corresponding to object positions.
        image = np.zeros((1, IMG_DIM, IMG_DIM))
        lowest_x, lowest_y = TAMPNAMOTransitionModel._state_to_world_frame(0, 0, 0)
        highest_x, highest_y = TAMPNAMOTransitionModel._state_to_world_frame(8, 1, 1)
        image_obj_size = int(OBJ_SIZE/min(highest_x, highest_y)*IMG_DIM)
        for room in range(9):
            objs_in_room = []
            for _ in range(self.num_obj_per_room if room < 8 else 1):
                if room == 8 or rand_state.rand() < INCLUSION_PROB:
                    obj_z = OBJ_HEIGHT/2  # include in room
                else:
                    obj_z = -OBJ_HEIGHT/2-0.05  # don't include in room
                while True:
                    obj_x = rand_state.uniform(0.2, 0.8)
                    obj_y = rand_state.uniform(0.2, 0.8)
                    if obj_z > 0 and \
                       any((obj_x-other[0])**2+(obj_y-other[1])**2 < 0.05
                               for other in objs_in_room):
                        continue
                    if room == 0 and obj_z > 0 and \
                       (obj_x-0.5)**2+(obj_y-0.5)**2 < 0.05:
                        continue
                    break
                if obj_z > 0:
                    objs_in_room.append((obj_x, obj_y))
                    world_x, world_y = TAMPNAMOTransitionModel._state_to_world_frame(room, obj_x, obj_y)
                    x_start = int(lin_interp(lowest_x, highest_x, 0, IMG_DIM, world_x))
                    y_start = int(lin_interp(lowest_y, highest_y, 0, IMG_DIM, world_y))
                    image[0, x_start:x_start+image_obj_size,
                          y_start:y_start+image_obj_size] = 1
                obj_pose_data.extend((obj_x, obj_y, obj_z))
        return np.array(obj_pose_data), image

    def heuristic(self, state):
        raise Exception("A* incompatible with a TAMP environment")

    @staticmethod
    def unflatten(action):
        """Convert the given flattened action into one that matches the
        form of the action space for this env.
        """
        assert len(action) == 15
        return [int(round(action[0])), action[1:8], action[8:]]


class TAMPNAMOProblemArbitrary(TAMPNAMOEnvFamily):
    """Example of a TAMP NAMO problem.
    """
    def __init__(self, generate_csi=True):
        super().__init__(generate_csi=generate_csi,
                         myseed=gc.rand_state.randint(1e8))


def create_tampnamo_env(base_class, myseed, global_context):
    """Method for dynamically creating a TAMPNAMO class.
    """
    name = "{}{}".format(base_class.__name__, myseed)
    def __init__(self, generate_csi=True):
        base_class.__init__(self, generate_csi=generate_csi, myseed=myseed)
    newclass = type(name, (base_class,), {"__init__" : __init__})
    global_context[newclass.__name__] = newclass
