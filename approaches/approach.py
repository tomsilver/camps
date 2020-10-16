"""An abstract interface for an approach.
"""

import abc


class Approach:
    """Abstract class defining an approach.
    """
    def __init__(self, solver):
        self._solver = solver

    @abc.abstractmethod
    def train(self, train_envs):
        """Train whatever you want, based on the given list of training
        environments. Doesn't return anything.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset_test_environment(self, test_env):
        """Set up whatever's needed for this new test environment. Return
        the planning_cost for this setup.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset_episode(self):
        """Set up whatever's needed for this new episode. Return
        the planning_cost for this setup.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_action(self, state):
        """Return a tuple of (action, planning_cost) for the current test
        environment and the given state. Here planning_cost just represents
        the cost for obtaining this action.
        """
        raise NotImplementedError("Override me!")
