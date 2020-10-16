"""An approach that transfers plans directly from training, using majority vote
at test time.
"""

import random
import pickle as pkl
from collections import Counter
import time
from approaches import Approach
from settings import EnvConfig as ec
from settings import ApproachConfig as ac
from utils import flatten


class PlanTransfer(Approach):
    """Plan transfer class definition.
    """
    def __init__(self, solver):
        super().__init__(solver)
        assert self._solver.is_online()  # else this approach makes no sense
        self._memorized_plans = None
        self._test_env = None
        self._step = None
        self._unflatten = None
        self._model_path = (ac.model_path_prefix+
                            self.__class__.__name__+
                            ".plans")

    def train(self, train_envs):
        self._memorized_plans = []
        for env in train_envs:
            print("Solving env {}".format(env.__class__.__name__), flush=True)
            pol = self._solver.solve(env, timeout=ec.train_solver_timeout)
            plan = []
            state = env.sample_initial_state()
            num_steps = 0
            while True:
                action = pol(state)
                if action is None:
                    break
                plan.append(action)
                state = env.sample_next_state(state, action)
                if env.reward(state, None)[1]:  # env is done
                    break
                if num_steps == ec.max_episode_length:
                    break
                num_steps += 1
            self._memorized_plans.append(plan)
        with open(self._model_path, "wb") as f:
            pkl.dump(self._memorized_plans, f)

    def reset_test_environment(self, test_env):
        self._test_env = test_env
        if self._memorized_plans is None:
            with open(self._model_path, "rb") as f:
                self._memorized_plans = pkl.load(f)
        if hasattr(test_env, "unflatten"):
            self._unflatten = test_env.unflatten
        return 0.0

    def reset_episode(self):
        self._step = 0
        return 0.0

    def get_action(self, state):
        start = time.time()
        suggestions = []
        for plan in self._memorized_plans:
            if self._step >= len(plan):
                # We've exhausted this plan, so ignore it.
                continue
            if self._unflatten is not None:
                suggestions.append(tuple(flatten(plan[self._step])))
            else:
                suggestions.append(plan[self._step])
        self._step += 1
        random.shuffle(suggestions)
        if not suggestions:  # all plans exhausted, take a random action
            action = self._test_env.action_var.sample()
        else:
            action = Counter(suggestions).most_common(1)[0][0]
            if action is not None and self._unflatten is not None:
                action = self._unflatten(action)
        step_cost = time.time()-start
        return action, step_cost
