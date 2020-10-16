"""Abstract interface for a solver.
"""

import abc


class Solver:
    """Base class for a solver.
    """
    def solve(self, env, timeout=None, vf=False):
        """solve() is any method that takes in an environment and an optional
        timeout, and returns a policy. A policy is a function that takes in a
        state and returns an action. If vf is True, return Q-values
        instead. Q-values are a map from (s, a) to a float."""
        info = env.get_solver_info()
        return self._solve(env, timeout, info, vf)

    @abc.abstractmethod
    def _solve(self, env, timeout=None, info=None, vf=False):
        """The actual _solve()"""
        raise NotImplementedError("Override me!")

    @staticmethod
    @abc.abstractmethod
    def is_online():
        """Return whether this solver is online.
        """
        raise NotImplementedError("Override me!")
