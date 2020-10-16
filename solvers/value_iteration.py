"""Value iteration solver.
"""

import time
from solvers import Solver
from settings import EnvConfig as ec
from settings import SolverConfig as sc


class ValueIteration(Solver):
    """Value iteration solver.
    """
    def _solve(self, env, timeout=None, info=None, vf=False):
        """Plan with value iteration. Requires full model of the world.
        Enumerates all states, so will not scale well.
        """
        qvals = self._vi_helper(env, timeout)
        if vf:
            return qvals
        def policy(state):
            return max(env.action_var.domain, key=lambda act: qvals[state, act])
        return policy

    @staticmethod
    def is_online():
        return False

    @staticmethod
    def _vi_helper(env, timeout):
        all_states = env.get_all_states()
        qvals = {(s, a): 0 for s in all_states for a in env.action_var.domain}
        itr = 0
        start = time.time()
        while True:
            new_qvals = qvals.copy()
            delta = 0
            for state in all_states:
                for act in env.action_var.domain:
                    rew, done = env.reward(state, act)
                    if done:
                        expec = 0
                    else:
                        expec = sum(prob*max(qvals[ns, na]
                                             for na in env.action_var.domain)
                                    for ns, prob in env.model(state, act))
                    new_qvals[state, act] = rew+ec.gamma*expec
                    delta = max(delta, abs(new_qvals[state, act]-
                                           qvals[state, act]))
                    if timeout is not None and time.time()-start > timeout:
                        return qvals
            qvals = new_qvals
            if delta < sc.vi_epsilon or itr == sc.vi_maxiters:
                return qvals
            itr += 1
