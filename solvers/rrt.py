"""RRT solver.
"""

import time
import numpy as np
from solvers import Solver
from structs import LIMBO
from settings import EnvConfig as ec
from settings import SolverConfig as sc
from settings import GeneralConfig as gc


class RRT(Solver):
    """Wrapper around BiRRT implementation that subclasses Solver.
    """
    def _solve(self, env, timeout=None, info=None, vf=False):
        if vf:
            raise NotImplementedError
        path = None
        clearance = ec.collision_clearance
        get_relevant_obstacle_vars = info["get_relevant_obstacle_vars"]
        get_updated_state = info["get_updated_state"]
        get_agent_position = info["get_agent_position"]
        sample_goal = info["sample_goal"]
        def policy(state):
            nonlocal path
            # Set up BiRRT
            def sample_fn(pt):
                _ = pt  # unused
                while True:
                    # Rejection sample until we find an action not in limbo.
                    action = env.action_var.sample()
                    updated_state = get_updated_state(state, action)
                    # Take a dummy action to update limbo.
                    updated_state = env.sample_next_state(
                        updated_state, action)
                    if LIMBO in updated_state and updated_state[LIMBO] == 1:
                        # This action would move us to limbo -> invalid.
                        continue
                    return action
            def extend_fn(pt1, pt2):
                pt1, pt2 = np.array(pt1), np.array(pt2)
                extension = []
                for mid in np.arange(0, 1+sc.extend_granularity,
                                     step=sc.extend_granularity):
                    extension.append(pt1+(pt2-pt1)*mid)
                for i, pt in enumerate(extension):
                    include_in_tree = (i == 0) or (i == len(extension) - 1)
                    yield pt, include_in_tree
            def collision_fn(pt):
                pt = np.array(pt)
                # First check whether we're in limbo.
                updated_state = get_updated_state(state, pt)
                # Take a dummy action to update limbo.
                updated_state = env.sample_next_state(updated_state, pt)
                if LIMBO in updated_state and updated_state[LIMBO] == 1:
                    return "limbo"
                # Now test collisions ONLY against obstacles in the
                # current region.
                for obs_var in get_relevant_obstacle_vars(pt):
                    rmin = state[obs_var[0]]
                    cmin = state[obs_var[1]]
                    rmax = state[obs_var[2]]
                    cmax = state[obs_var[3]]
                    if rmin-clearance <= pt[0] < rmax+clearance and \
                       cmin-clearance <= pt[1] < cmax+clearance:
                        return obs_var
                return None
            def distance_fn(pt1, pt2):
                pt1, pt2 = np.array(pt1), np.array(pt2)
                return np.linalg.norm(pt1-pt2)
            birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn)

            # Check whether there is any point on the existing path that we can reach
            if path is not None:
                current_pt = get_agent_position(state)
                for pt in path[::-1]:
                    # If we've reached this point, time to find a new path
                    if abs(pt[0] - current_pt[0]) + abs(pt[1] - current_pt[1]) < 1e-6:
                        path = None
                        break
                    direct_path = birrt._try_direct_path(current_pt, pt)
                    if direct_path is not None:
                        path = direct_path[2:]
                        # print("Continuing on direct path", path, "from", current_pt)
                        return path[0]

            # Run BiRRT.
            path = birrt.query(get_agent_position(state), sample_goal(state),
                               timeout=timeout)
            if path is not None:
                path = path[2:]  # strip out first b/c we're already there
                print("BiRRT found path of length {}".format(len(path)))
            if path is None:
                # BiRRT query failed, return a no-op.
                return get_agent_position(state)
            return path[0]
        return policy


class BiRRT:
    """Bidirectional rapidly-exploring random tree.
    """
    def __init__(self, sample_fn, extend_fn, collision_fn, distance_fn):
        self._sample_fn = sample_fn
        self._extend_fn = extend_fn
        self._collision_fn = collision_fn
        self._distance_fn = distance_fn

    def query(self, pt1, pt2, timeout):
        """Query the BiRRT, to get a collision-free path from pt1 to pt2.
        """
        if self._collision_fn(pt1) is not None or \
           self._collision_fn(pt2) is not None:
            return None
        direct_path = self._try_direct_path(pt1, pt2)
        if direct_path is not None:
            return direct_path
        start_time = time.time()
        for _ in range(sc.birrt_num_attempts):
            path = self._rrt_connect(pt1, pt2, start_time, timeout)
            if path is not None:
                return path
            if timeout is not None and time.time()-start_time > timeout:
                break
        return None

    def query_ignore_collisions(self, pt1, pt2):
        """Query the BiRRT but ignore collisions, thus returning a direct path.
        Also return the information for the first collision encountered.
        """
        path = [pt1]
        collision_info = self._collision_fn(pt1)
        if collision_info is None:
            collision_info = self._collision_fn(pt2)
        for newpt, flag in self._extend_fn(pt1, pt2):
            if collision_info is None:
                collision_info = self._collision_fn(newpt)
            if flag:
                path.append(newpt)
        return path, collision_info

    def _try_direct_path(self, pt1, pt2):
        path = [pt1]
        for newpt, flag in self._extend_fn(pt1, pt2):
            if self._collision_fn(newpt) is not None:
                return None
            if flag:
                path.append(newpt)
        return path

    def _rrt_connect(self, pt1, pt2, start_time, timeout):
        root1, root2 = TreeNode(pt1), TreeNode(pt2)
        nodes1, nodes2 = [root1], [root2]
        for _ in range(sc.birrt_num_iters):
            if len(nodes1) > len(nodes2):
                nodes1, nodes2 = nodes2, nodes1
            samp = self._sample_fn(pt1)
            nearest1 = min(nodes1, key=lambda n, samp=samp:
                           self._distance_fn(n.data, samp))
            for newpt, flag in self._extend_fn(nearest1.data, samp):
                if self._collision_fn(newpt) is not None:
                    break
                if flag:
                    nearest1 = TreeNode(newpt, parent=nearest1)
                    nodes1.append(nearest1)
            nearest2 = min(nodes2, key=lambda n, nearest1=nearest1:
                           self._distance_fn(n.data, nearest1.data))
            for newpt, flag in self._extend_fn(nearest2.data, nearest1.data):
                if self._collision_fn(newpt) is not None:
                    break
                if flag:
                    nearest2 = TreeNode(newpt, parent=nearest2)
                    nodes2.append(nearest2)
            else:
                path1 = nearest1.path_from_root()
                path2 = nearest2.path_from_root()
                if path1[0] != root1:
                    path1, path2 = path2, path1
                path = path1[:-1]+path2[::-1]
                return [node.data for node in path]
            if timeout is not None and time.time()-start_time > timeout:
                return None
        return None

    def smooth_path(self, path):
        """Return a smoothed path.
        """
        for _ in range(sc.birrt_smooth_amt):
            if len(path) <= 2:
                return path
            i = gc.rand_state.randint(0, len(path)-1)
            j = gc.rand_state.randint(0, len(path)-1)
            if abs(i-j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(newpt for newpt, _
                            in self._extend_fn(path[i], path[j]))
            if len(shortcut) < j-i and \
               all(self._collision_fn(pt) is None for pt in shortcut):
                path = path[:i+1]+shortcut+path[j+1:]
        return path

    @staticmethod
    def is_online():
        return True


class TreeNode:
    """TreeNode definition.
    """
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent

    def path_from_root(self):
        """Return the path from the root to this node.
        """
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]
