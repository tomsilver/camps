"""Implementation of task and motion planning (TAMP) bins
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


SKINNY_RAD = 0.05
FAT_RAD = 0.08
WIGGLE = 0.08
NUM_TARGETS = 2
NUM_DISTRACTORS = 15
BIN_OBJ_PROB = 0.5
BINS_DIM = 3
BIN_START_X = 0.75
BIN_Y = 0.9
BIN_START_Z = 0.68
BIN_SIZE = 0.25


BIN_LOCS = []
for dim_i in range(BINS_DIM):
    for dim_j in range(BINS_DIM):
        BIN_LOCS.append((BIN_START_X+BIN_SIZE*dim_i,
                         BIN_START_Z+BIN_SIZE*dim_j))


class TAMPBinsTransitionModel(TransitionModel):
    """Transition model for the TAMP bins environments.
    """
    def __init__(self, env):
        self._env = env
        self._constrained_env = None  # used for CSI learning
        self._p_targets = None
        self._p_distractors = None
        self._p_bin_objs = None
        self._p_all_objs = None
        self._p_held_obj_tf = None
        self._rrt = None
        self._last_grasp_style = None  # only for picking from the table
        self._initialize_pybullet()
        self._csilearning_data = None

    def get_state_vars(self):
        return self._env.state_variables

    def get_action_var(self, state_vars):
        return self._env.actvar

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
        - ("basecollision", <target object name>, <set of collisions>)
        - ("armcollision", <target object name>, <skill>, <set of collisions>)
        - ("needsregrasp", <target object name>)
        - ("unreached", <target object name>)
        - ("unmovable", <target object name>)
        """
        # Forcibly set pybullet state.
        self.force_pybullet_state(state)
        assert len(action) == 4
        skill, obj_ind, target_base_pose, target_grip_pose = action
        # Take action in pybullet.
        failure = self._update_pybullet_state(
            skill, obj_ind, target_base_pose, target_grip_pose)
        next_state_dict = state.todict()
        if skill == 0:  # movepick
            if np.linalg.norm(target_grip_pose[3:]-
                              np.array([0, 1, 0, 0])) < 0.01:
                next_state_dict[self._env.grasp_style] = 0  # top grasp
            elif np.linalg.norm(target_grip_pose[3:]-
                                np.array([-0.7071, 0, 0, 0.7071])) < 0.01:
                next_state_dict[self._env.grasp_style] = 1  # +y grasp
            elif np.linalg.norm(target_grip_pose[3:]-
                                np.array([0.7071, 0, 0, 0.7071])) < 0.01:
                next_state_dict[self._env.grasp_style] = 1
            elif np.linalg.norm(target_grip_pose[3:]-
                                np.array([0.5, 0.5, 0.5, 0.5])) < 0.01:
                next_state_dict[self._env.grasp_style] = 1
            elif np.linalg.norm(target_grip_pose[3:]-
                                np.array([0.5, -0.5, -0.5, 0.5])) < 0.01:
                next_state_dict[self._env.grasp_style] = 1
            # else:
            #     raise Exception("Unexpected grasp: {}".format(target_grip_pose))
        if skill == 2:  # moveplacebin
            for i, (x, z) in enumerate(BIN_LOCS):
                if x == target_grip_pose[0] and z == target_grip_pose[2]:
                    bin_idx = i
                    break
            else:
                next_state = state.state_factory.build(next_state_dict)
                return next_state, None
            next_state_dict[self._env.chosen_bin] = bin_idx
        else:
            next_state_dict[self._env.chosen_bin] = BINS_DIM*BINS_DIM
        if failure is not None:
            next_state = state.state_factory.build(next_state_dict)
            return next_state, failure  # failed; return failure info
        # Extract next-state information from pybullet.
        base_pos, base_orn = p.getBasePositionAndOrientation(self._p_robot)
        next_state_dict[self._env.r_base] = np.r_[base_pos, base_orn]
        grip_pos, grip_orn = p.getLinkState(self._p_robot, 6)[-2:]
        open_amt = p.getJointState(self._p_robot, 11)[0]
        next_state_dict[self._env.r_grip] = np.r_[grip_pos, grip_orn, open_amt]
        self._last_grasp_style = next_state_dict[self._env.grasp_style]
        if skill == 0:  # movepick
            next_state_dict[self._env.r_held] = obj_ind+1
        elif skill == 1:  # moveregrasp
            pass
            # assert next_state_dict[self._env.r_held] == obj_ind+1
        elif skill == 2:  # moveplacebin
            next_state_dict[self._env.r_held] = 0
        # else:
        #     raise Exception("Invalid skill: {}".format(skill))
        for obj, obj_loc, p_obj in zip(self._env.target_objs,
                                       self._env.target_obj_locs,
                                       self._p_targets):
            obj_pos, obj_orn = p.getBasePositionAndOrientation(p_obj)
            next_state_dict[obj] = np.r_[obj_pos, obj_orn]
            if obj_pos[0] < -0.2:
                next_state_dict[obj_loc] = 0  # table
            elif obj_pos[0] > 0.2:
                next_state_dict[obj_loc] = 1  # bin
            else:
                next_state_dict[obj_loc] = 2  # grasped
        next_state = state.state_factory.build(next_state_dict)
        return next_state, None

    def _generate_csilearning_data(self):
        assert BIN_OBJ_PROB == 0.0, "Need this for CSI learning"
        all_interesting_transitions = []
        all_interesting_actions = []
        for _ in range(25):
            env = TAMPBinsProblemArbitrary(generate_csi=False,
                                           dom_includes_side=True)
            state = env.sample_initial_state()
            num_tries = 0
            while True:
                action = env.action_var.sample()
                next_state, failure = self.sample_next_state_failure(
                    state, action)
                num_tries += 1
                if num_tries > 250:
                    break
                if self._env.reward(next_state, None)[1]:
                    break
                if failure is None:  # successful action
                    all_interesting_transitions.append((
                        state.todict(), action, next_state.todict()))
                    all_interesting_actions.append(action)
                    num_tries = 0  # reset
                    state = next_state
        self._csilearning_data = (all_interesting_transitions,
                                  all_interesting_actions)

    def get_random_constrained_transition(self, constraint):
        if self._csilearning_data is None:
            self._generate_csilearning_data()
        transitions, _ = self._csilearning_data
        while True:
            trans = transitions[gc.rand_state.randint(len(transitions))]
            state_dict, action, next_state_dict = trans
            state = self._env.state_factory.build(state_dict)
            next_state = self._env.state_factory.build(next_state_dict)
            if constraint.check(next_state):
                return state, action

    def update_constrained_transition(self, state, action, var):
        assert self._csilearning_data is not None
        _, actions = self._csilearning_data
        if var is self._env.action_var:
            action = actions[gc.rand_state.randint(len(actions))]
        else:
            if var.name.startswith("binobj"):  # try putting object in bin
                loc = BIN_LOCS[int(var.name[6:])]
                state = state.update(
                    var, np.r_[loc[0], BIN_Y, loc[1]-0.05, 0, 0, 0, 1])
            else:
                state = state.update(var, var.sample())
        return state, action

    def _set_targets_and_objs(self):
        obj_rad = self._env.reward.features[0]
        if obj_rad == SKINNY_RAD:
            self._p_targets = self._p_targets_skinny
            self._p_all_objs = self._p_all_objs_skinny
            for p_obj in self._p_targets_fat:  # move out of the way
                p.resetBasePositionAndOrientation(
                    p_obj, [0, 0, -100], [0, 0, 0, 1])
        elif obj_rad == FAT_RAD:
            self._p_targets = self._p_targets_fat
            self._p_all_objs = self._p_all_objs_fat
            for p_obj in self._p_targets_skinny:  # move out of the way
                p.resetBasePositionAndOrientation(
                    p_obj, [0, 0, -100], [0, 0, 0, 1])
        else:
            raise Exception("Invalid object radius: {}".format(obj_rad))

    def _initialize_pybullet(self):
        if p.getConnectionInfo()["isConnected"]:
            (self._p_robot, self._p_table, self._p_bin_table, self._p_bins,
             self._p_targets_skinny, self._p_targets_fat,
             self._p_distractors, self._p_bin_objs, self._p_all_objs_skinny,
             self._p_all_objs_fat, self._orig_bin_poses) = p_constants.PS_LIST
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
                                          [0, 0, 0.3],
                                          [0, 0, 0, 1])
        with open("envs/assets/table.urdf", "w") as fil:
            fil.write(p_constants.TABLE_URDF.format(
                1.2, 1.2, 1.2, 0.47, 0.47, 1))
        self._p_table = p.loadURDF("table.urdf", [-1, 1, -0.1])
        self._p_bin_table = p.loadURDF("table.urdf", [1, 1, -0.1])
        with open("envs/assets/back_plane.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                BIN_SIZE, 0.01, BIN_SIZE, 0.6, 0.3, 0.0, 1))
        with open("envs/assets/side_plane.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                0.01, BIN_SIZE, BIN_SIZE, 0.7, 0.4, 0.1, 1))
        with open("envs/assets/topbot_plane.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                BIN_SIZE, BIN_SIZE, 0.01, 0.7, 0.4, 0.1, 1))
        with open("envs/assets/blue_object.urdf", "w") as fil:
            fil.write(p_constants.CYLINDER_URDF.format(
                SKINNY_RAD, 0.15, 0, 0, 1, 1))
        with open("envs/assets/green_object.urdf", "w") as fil:
            fil.write(p_constants.CYLINDER_URDF.format(
                SKINNY_RAD, 0.15, 0, 1, 0, 1))
        with open("envs/assets/red_object.urdf", "w") as fil:
            fil.write(p_constants.CYLINDER_URDF.format(
                SKINNY_RAD, 0.15, 1, 0, 0, 1))
        with open("envs/assets/red_object_fat.urdf", "w") as fil:
            fil.write(p_constants.CYLINDER_URDF.format(
                FAT_RAD, 0.15, 1, 0, 0, 1))
        self._p_bins = []
        for x, z in BIN_LOCS:
            self._p_bins.append(p.loadURDF(
                "side_plane.urdf", [x-BIN_SIZE/2, BIN_Y, z]))
            self._p_bins.append(p.loadURDF(
                "side_plane.urdf", [x+BIN_SIZE/2, BIN_Y, z]))
            self._p_bins.append(p.loadURDF(
                "back_plane.urdf", [x, BIN_Y+BIN_SIZE/2, z]))
            self._p_bins.append(p.loadURDF(
                "topbot_plane.urdf", [x, BIN_Y, z-BIN_SIZE/2]))
            self._p_bins.append(p.loadURDF(
                "topbot_plane.urdf", [x, BIN_Y, z+BIN_SIZE/2]))
        self._orig_bin_poses = {}
        for p_bin in self._p_bins:
            self._orig_bin_poses[p_bin] = p.getBasePositionAndOrientation(p_bin)
        self._p_targets_skinny = []
        self._p_targets_fat = []
        for _ in self._env.target_objs:
            self._p_targets_skinny.append(p.loadURDF("red_object.urdf"))
            self._p_targets_fat.append(p.loadURDF("red_object_fat.urdf"))
        self._p_distractors = []
        for _ in self._env.distractors:
            self._p_distractors.append(p.loadURDF("blue_object.urdf"))
        self._p_bin_objs = []
        for _ in self._env.bin_objs:
            self._p_bin_objs.append(p.loadURDF("green_object.urdf"))
        self._p_all_objs_skinny = (self._p_targets_skinny+
                                   self._p_distractors+
                                   self._p_bin_objs)
        self._p_all_objs_fat = (self._p_targets_fat+
                                self._p_distractors+
                                self._p_bin_objs)
        for fil in glob.glob("envs/assets/table.urdf"):
            os.remove(fil)
        for fil in glob.glob("envs/assets/*_plane*.urdf"):
            os.remove(fil)
        for fil in glob.glob("envs/assets/*object*.urdf"):
            os.remove(fil)
        p_constants.set_ps([self._p_robot, self._p_table, self._p_bin_table,
                            self._p_bins, self._p_targets_skinny,
                            self._p_targets_fat, self._p_distractors,
                            self._p_bin_objs, self._p_all_objs_skinny,
                            self._p_all_objs_fat, self._orig_bin_poses])
        self._setup_motion_planner()

    def _setup_motion_planner(self):
        def _sample_fn(_):
            corners = [[-1.9, 0.1], [-1.9, 1.9],
                       [0, 0.1], [0, 1.9],
                       [1.9, 0.1], [1.9, 1.9]]
            return corners[gc.rand_state.randint(len(corners))]
        def _extend_fn(pt1, pt2):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            num = int(np.ceil(max(abs(pt1-pt2))))*10
            if num == 0:
                yield pt2, True
            for i in range(1, num+1):
                yield np.r_[pt1*(1-i/num)+pt2*i/num], True
        def _collision_fn(pt):
            x, y = pt
            if -1.75 < x < -0.25 and 0.25 < y < 1.75:
                return "table"
            if 0.25 < x < 1.75 and 0.25 < y < 1.75:
                return "bin_table"
            return None
        def _distance_fn(pt1, pt2):
            return (pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2
        self._rrt = BiRRT(_sample_fn, _extend_fn, _collision_fn, _distance_fn)

    def force_pybullet_state(self, state):
        """Forcibly set pybullet state to the given one.
        """
        self._set_targets_and_objs()
        self._grab_object(obj_id=None)
        # Reset bin poses.
        for p_ind, pose in self._orig_bin_poses.items():
            p.resetBasePositionAndOrientation(p_ind, pose[0], pose[1])
        # Set base pose.
        r_base = state[self._env.r_base]
        cur_z = p.getBasePositionAndOrientation(self._p_robot)[0][2]
        target_orn = p.getEulerFromQuaternion(r_base[3:])[2]
        self._move_base(np.r_[r_base[:2], cur_z], target_orn,
                        do_interpolate=False)
        # Set object poses.
        for obj, p_obj in zip(self._env.target_objs, self._p_targets):
            pose = state[obj]
            p.resetBasePositionAndOrientation(p_obj, pose[:3], pose[3:])
        for dist, p_dist in zip(self._env.distractors, self._p_distractors):
            pose = state[dist]
            p.resetBasePositionAndOrientation(p_dist, pose[:3], pose[3:])
        for obj, p_obj in zip(self._env.bin_objs, self._p_bin_objs):
            pose = state[obj]
            p.resetBasePositionAndOrientation(p_obj, pose[:3], pose[3:])
        # Set gripper pose.
        r_grip = state[self._env.r_grip]
        success = self._move_end_effector(
            r_grip[:3], r_grip[3:7], do_interpolate=False, open_amt=0)#r_grip[7])
        _ = success  # ignore
        # assert success, "Setting state should never fail..."
        # Handle grabbing object, if one is held.
        held = state[self._env.r_held]
        if held == 0:
            self._grab_object(obj_id=None)
        else:
            self._grab_object(obj_id=self._p_targets[held-1])

    def _update_pybullet_state(self, skill, obj_ind, target_base_pose,
                               target_grip_pose):
        """Returns None if updating state was successful, and otherwise returns
        the reason for failure.
        """
        if obj_ind >= len(self._env.all_objs) or obj_ind < 0:
            return None
        target_obj_name = self._env.all_objs[obj_ind].name
        if target_obj_name.startswith("distractor"):
            return ("unmovable", target_obj_name)
        if skill == 2:
            obj_orn = p.getBasePositionAndOrientation(
                self._p_all_objs[obj_ind])[1][-1]
            if obj_orn > 0.7:  # currently top-grasped
                return ("needsregrasp", target_obj_name)
        r_base = target_base_pose
        do_interpolate = (p.getConnectionInfo()["connectionMethod"] == p.GUI and
                          gc.do_render)
        orig_grip_pos, orig_grip_orn = p.getLinkState(self._p_robot, 6)[-2:]
        # Handle base motion.
        cur_xy = p.getBasePositionAndOrientation(self._p_robot)[0][:2]
        path = self._rrt.query(cur_xy, r_base[:2], timeout=30)
        if path is None:
            _, cols = self._rrt.query_ignore_collisions(cur_xy, r_base[:2])
            return ("basecollision", target_obj_name, cols)
        (_, _, cur_z), cur_orn = p.getBasePositionAndOrientation(
            self._p_robot)
        cur_orn = p.getEulerFromQuaternion(cur_orn)[2]
        target_orn = p.getEulerFromQuaternion(r_base[3:])[2]
        for i, loc in enumerate(path):
            interp_orn = cur_orn*(1-i/len(path))+target_orn*i/len(path)
            self._move_base(np.r_[loc, cur_z], interp_orn, do_interpolate)
        # Handle gripper motion.
        if skill == 1:
            # Handle special regrasping motions.
            r_grip = np.copy(target_grip_pose)
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
            self._grab_object(None)  # let go
            r_grip[2] += 0.2
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
            r_grip[0] -= 0.40
            r_grip[3:7] = [0.5, 0.5, 0.5, 0.5]
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
            r_grip[2] = 0.63
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
            r_grip[0] += 0.15
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
            self._grab_object(self._p_all_objs[obj_ind])  # pick up
        elif skill == 2:
            # Handle special placing in bin motions.
            r_grip = np.copy(target_grip_pose)
            r_grip[1] -= 0.2
            r_grip[2] -= 0.05
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
            r_grip[1] += 0.2
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
        else:
            r_grip = np.copy(target_grip_pose)
            success = self._move_end_effector(r_grip[:3], r_grip[3:7],
                                              do_interpolate, open_amt=0)
            if not success:
                return ("unreached", target_obj_name)
        # Do collision-checking.
        p.stepSimulation()
        cols = self._get_robot_collisions()
        if self._env.all_objs[obj_ind].name not in cols:
            return ("unreached", target_obj_name)  # we WANT to be colliding
        cols.remove(self._env.all_objs[obj_ind].name)
        cols.discard("table_bin")
        if cols:  # there's some other collision
            return ("armcollision", target_obj_name, skill, cols)
        # Handle grabbing or letting go of object.
        if skill == 0:  # movepick
            self._grab_object(self._p_all_objs[obj_ind])
        elif skill == 1:  # moveregrasp
            pass
        elif skill == 2:  # moveplacebin
            self._grab_object(None)
        # else:
        #     raise Exception("Invalid skill: {}".format(skill))
        if skill == 0:
            r_grip[2] += 0.2
            self._move_end_effector(r_grip[:3], r_grip[3:7],
                                    do_interpolate, open_amt=0)
        # Restore base pose.
        for i, loc in enumerate(path[::-1]):
            interp_orn = target_orn*(1-i/len(path))+cur_orn*i/len(path)
            self._move_base(np.r_[loc, cur_z], interp_orn, do_interpolate)
        # Restore gripper pose.
        self._move_end_effector(orig_grip_pos, orig_grip_orn,
                                do_interpolate, open_amt=0)
        return None  # no failure

    def _get_robot_collisions(self):
        cols = set()
        for cp in p.getContactPoints(self._p_robot):
            assert cp[1] == self._p_robot
            if cp[2] == self._p_table:
                cols.add("table")
            elif cp[2] == self._p_bin_table:
                cols.add("table_bin")
            elif cp[2] in self._p_bins:
                cols.add("binwall")
            else:
                assert cp[2] in self._p_all_objs, "Unexpected col: {}".format(
                    cp[2])
                cols.add(self._env.all_objs[self._p_all_objs.index(cp[2])].name)
        return cols

    def get_p_targets(self):
        """Get pybullet IDs of target objects.
        """
        return self._p_targets

    def get_p_all_objs(self):
        """Get pybullet IDs of all objects.
        """
        return self._p_all_objs

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

    def _move_end_effector(self, target_loc, target_orn, do_interpolate,
                           open_amt=None, avoid_cols=False, succ_thresh=0.05):
        num_iter = 0
        while True:
            num_iter += 1
            if num_iter > 10000:
                return False
            njo = min(p.getNumJoints(self._p_robot), 7)
            old_joints = np.array([p.getJointState(self._p_robot,
                                                   joint_idx)[0]
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


class TAMPBinsEnvFamily(Environment):
    """TAMP bins environment family definition.
    """
    se3_bounds = [np.ones(7)*-100, np.ones(7)*100]
    se3_bounds_plus_one = [np.ones(8)*-100, np.ones(8)*100]
    TABLE_BASE_POSES = [
        [-1.4, 0.25, 0, 0, 0, 0, 1],
        [-1.0, 0.25, 0, 0, 0, 0, 1],
        [-0.6, 0.25, 0, 0, 0, 0, 1],
        [-1.8, 0.7, 0, 0, 0, -0.7071, 0.7071],
        [-1.8, 1.1, 0, 0, 0, -0.7071, 0.7071],
        [-1.8, 1.4, 0, 0, 0, -0.7071, 0.7071],
        [-1.4, 1.75, 0, 0, 0, 1, 0],
        [-1.0, 1.75, 0, 0, 0, 1, 0],
        [-0.6, 1.75, 0, 0, 0, 1, 0],
        [-0.25, 0.7, 0, 0, 0, 0.7071, 0.7071],
        [-0.25, 1.1, 0, 0, 0, 0.7071, 0.7071],
        [-0.25, 1.4, 0, 0, 0, 0.7071, 0.7071],
    ]
    temp_x = 0.5
    robot_y = 0.05#-0.15
    TEMP_BASE_POSES = [[temp_x, robot_y, 0, 0, 0, 0, 1]]
    bin_xs = set(loc[0] for loc in BIN_LOCS)
    assert len(bin_xs) == BINS_DIM
    BIN_BASE_POSES = {}
    for bin_x in bin_xs:
        BIN_BASE_POSES[bin_x] = [bin_x, robot_y, 0, 0, 0, 0, 1]
    ALL_BASE_POSES = (TABLE_BASE_POSES+
                      TEMP_BASE_POSES+
                      list(BIN_BASE_POSES.values()))
    # Action: skill (move-pick 0 or move-regrasp 1 or move-placebin 2),
    # object to act upon, target base pose, and target gripper pose.
    ACT_SKILL = DiscreteVariable("act_skill", 3)
    ACT_OBJ = DiscreteVariable("act_obj", NUM_TARGETS+NUM_DISTRACTORS)
    ACT_BASE = ContinuousVariable("act_base", se3_bounds)
    ACT_GRIP = ContinuousVariable("act_grip", se3_bounds)
    state_variables = []
    # Robot base pose.
    r_base = ContinuousStateVariable("robotbase", se3_bounds)
    # Robot end effector pose, plus a final dimension for gripper open amount.
    r_grip = ContinuousStateVariable("robotgrip", se3_bounds_plus_one)
    # Grasp style (top 0 or side 1). Used for CSIs.
    # Only used for picking from table.
    # Picking from temp location is always done as a side grasp.
    grasp_style = StateVariable("graspstyle", 2)
    state_variables.extend([r_base, r_grip, grasp_style])
    # Held object index (0 means none held).
    r_held = StateVariable("robotheld", NUM_TARGETS+1)
    state_variables.append(r_held)
    # Target object poses.
    target_objs = []
    for i in range(NUM_TARGETS):
        obj = ContinuousStateVariable("targetobj{}".format(i), se3_bounds)
        target_objs.append(obj)
        state_variables.append(obj)
    # Where are the target objects? 0 = table, 1 = bin, 2 = grasped.
    # Used for specifying goal.
    target_obj_locs = []
    for i in range(NUM_TARGETS):
        obj_loc = StateVariable("targetobj{}loc".format(i), 3)
        target_obj_locs.append(obj_loc)
        state_variables.append(obj_loc)
    # Distractor object poses.
    distractors = []
    for i in range(NUM_DISTRACTORS):
        dist = ContinuousStateVariable("distractor{}".format(i), se3_bounds)
        distractors.append(dist)
        state_variables.append(dist)
    # Bin object poses.
    bin_objs = []
    for i in range(BINS_DIM*BINS_DIM):
        obj = ContinuousStateVariable("binobj{}".format(i), se3_bounds)
        bin_objs.append(obj)
        state_variables.append(obj)
    all_objs = target_objs+distractors+bin_objs
    # Chosen bin to use. Last index is initial value (unused after init state).
    chosen_bin = StateVariable("chosenbin", BINS_DIM*BINS_DIM+1)
    state_variables.append(chosen_bin)
    state_factory = StateFactory(state_variables)

    def __init__(self, generate_csi=True, dom_includes_side=False, myseed=None):
        self._myseed = myseed
        (self._initial_state,
         self._goal,
         self._goal_pddl) = self._construct_initstate_and_goal()
        reward_fn = self._construct_reward_fn()
        self.reward = reward_fn
        transition_model = TAMPBinsTransitionModel(self)
        transition_model.force_pybullet_state(self._initial_state)
        act_domain = []
        all_grip_poses = []
        for obj in transition_model.get_p_targets():
            op = p.getBasePositionAndOrientation(obj)
            # NOTE: should match below (1)
            if self.reward.features[0] == FAT_RAD:
                all_grip_poses.extend([
                    [op[0][0], op[0][1]-0.3, 0.66, -0.7071, 0, 0, 0.7071],  # +y
                    [op[0][0], op[0][1]+0.3, 0.66, 0.7071, 0, 0, 0.7071],  # -y
                    [op[0][0]-0.3, op[0][1], 0.66, 0.5, 0.5, 0.5, 0.5],  # +x
                    [op[0][0]+0.3, op[0][1], 0.66, 0.5, -0.5, -0.5, 0.5],  # -x
                ])
            elif dom_includes_side:
                all_grip_poses.extend([
                    [op[0][0], op[0][1], 0.9, 0, 1, 0, 0],  # top
                    [op[0][0], op[0][1]-0.3, 0.66, -0.7071, 0, 0, 0.7071],  # +y
                    [op[0][0], op[0][1]+0.3, 0.66, 0.7071, 0, 0, 0.7071],  # -y
                    [op[0][0]-0.3, op[0][1], 0.66, 0.5, 0.5, 0.5, 0.5],  # +x
                    [op[0][0]+0.3, op[0][1], 0.66, 0.5, -0.5, -0.5, 0.5],  # -x
                ])
            else:
                all_grip_poses.extend([
                    [op[0][0], op[0][1], 0.9, 0, 1, 0, 0],  # top
                ])
        # Add temp pose.
        # NOTE: should match below (2)
        all_grip_poses.append([self.temp_x, 0.6, 0.9, 0, 1, 0, 0])
        # Add bin place poses.
        # NOTE: should match below (3)
        for x, z in BIN_LOCS:
            all_grip_poses.append([x, BIN_Y-0.3, z, -0.7071, 0, 0, 0.7071])
        for val1 in self.ACT_SKILL.domain:
            for val2 in self.ACT_OBJ.domain:
                if val2 >= NUM_TARGETS:
                    continue
                for val3 in self.ALL_BASE_POSES:
                    for val4 in all_grip_poses:
                        act_domain.append([val1, val2, val3, val4])
        self.actvar = MultiDimVariable("action", [self.ACT_SKILL, self.ACT_OBJ,
                                                  self.ACT_BASE, self.ACT_GRIP],
                                       act_domain)
        super().__init__(transition_model, reward_fn, self.state_factory,
                         generate_csi=generate_csi)

    def get_solver_info(self, relaxation=None):
        """This method gives special methods needed by the TAMP solver.
        """
        info = {}

        def _get_domain_pddl():
            return """
            (define (domain tampbinsdomain)
              (:requirements :strips :typing)
              (:types object pose)
              (:predicates
                (at ?o - object ?p - pose)
                (needsregrasp ?o - object)
                (isgp ?bp - pose ?gp - pose ?o - object ?op - pose) ; isgrasppose
                (istp ?bp - pose ?gp - pose ?o - object ?tp - pose) ; istemppose
                (issp ?bp - pose ?gp - pose ?o - object ?tp - pose) ; isstowpose
                (objobstructs ?o - object ?p - pose)
                (emptygripper)
                (ingripper ?o - object)
                (clear ?p - pose)
                (unmovable ?o - object)
              )
              (:action movepick
                :parameters (?o - object ?bp - pose ?gp - pose ?op - pose)
                :precondition (and (at ?o ?op)
                                   (isgp ?bp ?gp ?o ?op)
                                   (forall (?o2 - object) (not (objobstructs ?o2 ?gp)))
                                   (emptygripper)
                                   (not (unmovable ?o))
                                   (not (clear ?op))
                              )
                :effect (and (not (at ?o ?op))
                             (not (emptygripper))
                             (ingripper ?o)
                             (clear ?op)
                             (forall (?p2 - pose) (not (objobstructs ?o ?p2)))
                        )
              )
              (:action moveregrasp
                :parameters (?o - object ?bp - pose ?gp - pose ?tp - pose)
                :precondition (and (ingripper ?o)
                                   (needsregrasp ?o)
                                   (forall (?o2 - object) (not (objobstructs ?o2 ?gp)))
                                   (clear ?tp)
                                   (istp ?bp ?gp ?o ?tp)
                              )
                :effect (and (not (needsregrasp ?o))
                        )
              )
              (:action moveplacebin
              :parameters (?o - object ?bp - pose ?gp - pose ?tp - pose)
                :precondition (and (ingripper ?o)
                                   (issp ?bp ?gp ?o ?tp)
                                   (forall (?o2 - object) (not (objobstructs ?o2 ?gp)))
                                   (clear ?tp)
                                   (not (needsregrasp ?o))
                              )
                :effect (and (at ?o ?tp)
                             (emptygripper)
                             (not (ingripper ?o))
                             (not (clear ?tp))
                        )
              )
            )"""
        info["get_domain_pddl"] = _get_domain_pddl

        def _get_problem_pddl(state, discovered_facts):
            state = state.todict()
            state_pddl = []
            objects = []
            for key, val in state.items():
                if key in self.target_obj_locs or key in self.distractors:
                    if key in self.target_obj_locs:
                        name = key.name[:-3]
                    else:
                        name = key.name
                    objects.append("{} - object".format(name))
                    objects.append("bp_{}_table - pose".format(name))
                    objects.append("gp_{}_table - pose".format(name))
                    objects.append("op_{}_table - pose".format(name))
                    state_pddl.append("(isgp bp_{0}_table gp_{0}_table {0} "
                                      "op_{0}_table)".format(name))
                    objects.append("bp_{}_temp - pose".format(name))
                    objects.append("gp_{}_temp - pose".format(name))
                    objects.append("tp_{}_temp - pose".format(name))
                    state_pddl.append("(istp bp_{0}_temp gp_{0}_temp {0} "
                                      "tp_{0}_temp)".format(name))
                    if key in self.distractors:
                        continue
                    objects.append("bp_{}_bins - pose".format(name))
                    objects.append("gp_{}_bins - pose".format(name))
                    objects.append("tp_{}_bins - pose".format(name))
                    state_pddl.append("(issp bp_{0}_bins gp_{0}_bins {0} "
                                      "tp_{0}_bins)".format(name))
                    if val == 0:  # table
                        state_pddl.append("(at {0} op_{0}_table)".format(name))
                        state_pddl.append("(clear tp_{}_temp)".format(name))
                        state_pddl.append("(clear tp_{}_bins)".format(name))
                    elif val == 1:  # bin
                        state_pddl.append("(at {0} tp_{0}_bins)".format(name))
                        state_pddl.append("(clear op_{}_table)".format(name))
                        state_pddl.append("(clear tp_{}_temp)".format(name))
                    elif val == 2:  # grasped
                        state_pddl.append("(clear op_{}_table)".format(name))
                        state_pddl.append("(clear tp_{}_temp)".format(name))
                        state_pddl.append("(clear tp_{}_bins)".format(name))
                    else:
                        raise Exception("Unexpected val: {}".format(val))
                if key == self.r_held:
                    if val == 0:
                        state_pddl.append("(emptygripper)")
                    else:
                        name = self.target_objs[val-1]
                        state_pddl.append("(ingripper {})".format(name))
            state_pddl.extend(discovered_facts)
            objects = "\n\t".join(objects)
            state_pddl = "\n\t".join(state_pddl)
            return """
            (define (problem tampbinsproblem)
              (:domain tampbinsdomain)
              (:objects\n\t{}
              )
              (:goal {})
              (:init\n\t{}
              )
            )""".format(objects, self._goal_pddl, state_pddl)
        info["get_problem_pddl"] = _get_problem_pddl

        def _refine_step(state, symbolic_action):
            state_dict = state.todict()
            act = symbolic_action.split()
            obj_ind = [i for i, o in enumerate(self.all_objs)
                       if o.name == act[1]][0]
            if act[0] == "movepick":
                # Randomly select a base pose.
                base_poses = self.TABLE_BASE_POSES
                base_pose = base_poses[gc.rand_state.randint(len(base_poses))]
                # Randomly select a grasp pose.
                op = p.getBasePositionAndOrientation(
                    self.transition_model.get_p_all_objs()[obj_ind])
                # NOTE: should match above (1)
                grip_poses = [
                    [op[0][0], op[0][1], 0.9, 0, 1, 0, 0],  # top
                    [op[0][0], op[0][1]-0.3, 0.66, -0.7071, 0, 0, 0.7071],  # +y
                    [op[0][0], op[0][1]+0.3, 0.66, 0.7071, 0, 0, 0.7071],  # -y
                    [op[0][0]-0.3, op[0][1], 0.66, 0.5, 0.5, 0.5, 0.5],  # +x
                    [op[0][0]+0.3, op[0][1], 0.66, 0.5, -0.5, -0.5, 0.5],  # -x
                ]
                if self.reward.features[0] == FAT_RAD:
                    grip_poses.pop(0)
                grip_pose = grip_poses[gc.rand_state.randint(len(grip_poses))]
                action = [0, obj_ind, base_pose, grip_pose]
            if act[0] == "moveregrasp":
                base_poses = self.TEMP_BASE_POSES
                assert len(base_poses) == 1
                base_pose = base_poses[0]
                # NOTE: should match above (2)
                grip_pose = [self.temp_x, 0.6, 0.9, 0, 1, 0, 0]
                action = [1, obj_ind, base_pose, grip_pose]
            if act[0] == "moveplacebin":
                # Randomly select a bin. Check against constraint.
                if relaxation is not None:
                    constraint = relaxation[0].imposed_constraint
                    num_tries = 0
                    while True:
                        bin_idx = gc.rand_state.randint(len(BIN_LOCS))
                        state_dict[self.chosen_bin] = bin_idx
                        num_tries += 1
                        if num_tries > 1000:
                            break
                        if constraint.check(state_dict):
                            break
                else:
                    bin_idx = gc.rand_state.randint(len(BIN_LOCS))
                x, z = BIN_LOCS[bin_idx]
                base_pose = self.BIN_BASE_POSES[x]
                # NOTE: should match below (3)
                grip_pose = [x, BIN_Y-0.3, z, -0.7071, 0, 0, 0.7071]
                action = [2, obj_ind, base_pose, grip_pose]
            return action
        info["refine_step"] = _refine_step

        def _failure_to_facts(failure):
            if failure[0] in ("basecollision", "unreached"):
                return None  # no fact to discover, just a bad refinement
            if failure[0] == "unmovable":
                return ["(unmovable {})".format(failure[1])]
            if failure[0] == "armcollision":
                target, skill, cols = failure[1], failure[2], failure[3]
                if skill == 0:
                    pose_suffix = "table"
                elif skill == 1:
                    pose_suffix = "temp"
                elif skill == 2:
                    pose_suffix = "bins"
                else:
                    raise Exception("Unexpected skill: {}".format(skill))
                facts = []
                for col in cols:
                    if col.startswith("table"):
                        continue
                    if col == "binwall":
                        continue
                    if col.startswith("binobj"):  # just a bad refinement
                        continue
                    assert "distractor" not in target
                    facts.append("(objobstructs {} gp_{}_{})".format(
                        col, target, pose_suffix))
                if not facts:
                    return None
                return facts
            if failure[0] == "needsregrasp":
                return ["(needsregrasp {})".format(failure[1])]
            raise Exception("Unexpected failure: {}".format(failure))
        info["failure_to_facts"] = _failure_to_facts

        return info

    def sample_initial_state(self):
        self.transition_model.force_pybullet_state(self._initial_state)
        return self._initial_state

    def render(self, state):
        pass

    def _construct_reward_fn(self):
        reward_vars = self.target_obj_locs+[self.r_base, self.r_grip, self.r_held]
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
        state_dict[self.r_base] = [-0.1, 0, 0, 0, 0, 0, 1]
        orn = p.getQuaternionFromEuler((0, np.pi, 0))
        state_dict[self.r_grip] = np.r_[[0, 0.3, 1], orn, [0]]
        state_dict[self.grasp_style] = 0
        state_dict[self.r_held] = 0
        obj_pose_data, _ = self._get_reward_features()
        assert len(self.all_objs) == len(obj_pose_data)
        for obj in self.all_objs:
            pose = obj_pose_data[obj]
            state_dict[obj] = pose
        for loc in self.target_obj_locs:
            state_dict[loc] = 0  # init: targets on table
        state_dict[self.chosen_bin] = BINS_DIM*BINS_DIM
        init_state = self.state_factory.build(state_dict)
        goal = lambda state: all(state[obj_loc] == 1  # goal: targets in bin
                                 for obj_loc in self.target_obj_locs)
        goal_pddl = []
        for obj in self.target_objs:
            goal_pddl.append("(at {0} tp_{0}_bins)".format(obj.name))
        goal_pddl = "(and {})".format(" ".join(goal_pddl))
        return init_state, goal, goal_pddl

    def _get_reward_features(self):
        rand_state = np.random.RandomState(seed=self._myseed)
        obj_pose_data = {}
        features = []
        table_coords = np.vstack(list(map(np.ravel, np.meshgrid(
            np.arange(-1-0.45, -1+0.5, 0.2), np.arange(1-0.45, 1+0.5, 0.2))))).T
        table_coords = sorted(table_coords,
                              key=lambda x: (x[0]+1)**2+(x[1]-1)**2,
                              reverse=True)
        first_few = np.array(table_coords[:10])
        remaining = np.array(table_coords[10:])
        rand_state.shuffle(first_few)
        rand_state.shuffle(remaining)
        obj_rad = SKINNY_RAD if rand_state.rand() < 0.5 else FAT_RAD
        features.append(obj_rad)
        table_coords = np.r_[first_few, remaining]
        table_coords_ind = 0
        obj_z = 0.615
        for obj in self.target_objs:
            coords = table_coords[table_coords_ind]
            noise = rand_state.uniform(-WIGGLE, WIGGLE, size=coords.shape)
            obj_pose_data[obj] = np.r_[coords+noise, obj_z, 0, 0, 0, 1]
            table_coords_ind += 1
        dist_z = obj_z
        for dist in self.distractors:
            coords = table_coords[table_coords_ind]
            noise = rand_state.uniform(-WIGGLE, WIGGLE, size=coords.shape)
            obj_pose_data[dist] = np.r_[coords+noise, dist_z, 0, 0, 0, 1]
            table_coords_ind += 1
        includeds = np.zeros(len(BIN_LOCS))
        for i, (obj, loc) in enumerate(zip(self.bin_objs, BIN_LOCS)):
            if rand_state.rand() < BIN_OBJ_PROB:
                obj_pose_data[obj] = np.r_[loc[0], BIN_Y, loc[1]-0.05,
                                           0, 0, 0, 1]
                includeds[i] = 1
            else:  # don't include object in bin
                obj_pose_data[obj] = np.r_[loc[0], BIN_Y, -10,
                                           0, 0, 0, 1]
        features.extend(includeds)
        return obj_pose_data, features

    @staticmethod
    def unflatten(action):
        """Convert the given flattened action into one that matches the
        form of the action space for this env.
        """
        assert len(action) == 16
        return [int(round(action[0])), int(round(action[1])), action[2:9], action[9:]]

    def heuristic(self, state):
        raise Exception("A* incompatible with a TAMP environment")


class TAMPBinsProblemArbitrary(TAMPBinsEnvFamily):
    """Example of a TAMP bins problem.
    """
    def __init__(self, generate_csi=True, dom_includes_side=False):
        super().__init__(generate_csi=generate_csi,
                         dom_includes_side=dom_includes_side,
                         myseed=gc.rand_state.randint(1e8))


def create_tampbins_env(base_class, myseed, global_context):
    """Method for dynamically creating a TAMPBins class.
    """
    name = "{}{}".format(base_class.__name__, myseed)
    def __init__(self, generate_csi=True, dom_includes_side=False):
        base_class.__init__(self, generate_csi=generate_csi,
                            dom_includes_side=dom_includes_side, myseed=myseed)
    newclass = type(name, (base_class,), {"__init__" : __init__})
    global_context[newclass.__name__] = newclass
