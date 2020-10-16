"""An approach that plans on the full test environment. Does nothing
during training.
"""

import time
from approaches import Approach
from settings import EnvConfig as ec


class FullTestPlanner(Approach):
    """Full test planner class definition.
    """
    def __init__(self, solver):
        super().__init__(solver)
        self._test_policy = None
        self._solver_timeout = ec.test_solver_timeout

    def train(self, train_envs):
        pass

    def reset_test_environment(self, test_env):
        start = time.time()
        self._test_policy = self._solver.solve(
            test_env, timeout=self._solver_timeout)
        solve_cost = time.time()-start
        return solve_cost

    def reset_episode(self):
        return 0.0

    def get_action(self, state):
        assert self._test_policy is not None, "Did you reset first?"
        start = time.time()
        action = self._test_policy(state)
        step_cost = time.time()-start
        return action, step_cost
