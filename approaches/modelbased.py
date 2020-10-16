"""Approach that trains a constraint predictive model on the train envs.
Uses this model to pick a constraint during test time.
"""

import pickle as pkl
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from termcolor import colored
from settings import ApproachConfig as ac
from settings import EnvConfig as ec
from settings import GeneralConfig as gc
from approaches import Approach
from structs import LIMBO, StateFactory
from envs.env_base import RewardFunction, Environment, TransitionModel
from utils import test_approach, FCN, CNN


class ModelBased(Approach):
    """ModelBased class definition with variable dropping.
    """
    def __init__(self, solver):
        super().__init__(solver)
        self._constraint_model = None
        self._drop_variables = True
        self._use_limbo = True
        self._test_policy = None
        self._constraint_to_use = None
        self._solver_timeout = ec.test_solver_timeout
        self._train_candidate_constraints = None
        self._model_path = (ac.model_path_prefix+
                            self.__class__.__name__+
                            ".model")

    def train(self, train_envs):
        # Temporarily set solver timeout for training
        original_solver_timeout = self._solver_timeout
        self._solver_timeout = ec.train_solver_timeout

        candidate_constraints = self._sorted_candidate_constraints(train_envs[0])
        self._train_candidate_constraints = candidate_constraints
        X = []
        Y = []
        theta_shape = None
        self._data_cache = {}
        for env in train_envs:
            print("Training on env {}".format(env.__class__.__name__), flush=True)
            assert env.reward.features is not None
            theta = np.array(env.reward.features)
            best_constraint = max(candidate_constraints,
                                  key=lambda constr: self._score_constraint(
                                      constr, env))
            if ec.net_arch == "FCN":
                X.append(theta)
            elif ec.net_arch == "CNN":
                if theta_shape is None:
                    theta_shape = theta.shape
                else:  # all theta must be the same shape
                    assert theta_shape == theta.shape
                X.append(theta.flatten())
            else:
                raise Exception("Unrecognized net_arch: {}".format(ec.net_arch))
            Y.append(candidate_constraints.index(best_constraint))
            np.save("{}.X.npy".format(self._model_path),
                    np.array(X, dtype=np.float32))
            np.save("{}.Y.npy".format(self._model_path),
                    np.array(Y, dtype=np.int))
        # X = np.load("{}.X.npy".format(self._model_path))
        # Y = np.load("{}.Y.npy".format(self._model_path))
        # theta_shape = np.array(train_envs[0].reward.features).shape
        X = torch.from_numpy(np.array(X, dtype=np.float32))
        Y = torch.from_numpy(np.array(Y, dtype=np.int))
        if ec.net_arch == "FCN":
            self._constraint_model = FCN(in_size=X.shape[1],
                                         hid_sizes=[50, 32, 10],
                                         out_size=len(candidate_constraints),
                                         do_softmax=True)
        elif ec.net_arch == "CNN":
            do_max_pool = (ec.family_to_run == "tampnamo")
            self._constraint_model = CNN(in_size=X.shape[1], num_channels=10,
                                         kernel_size=2, hid_sizes=[32, 10],
                                         theta_shape=theta_shape,
                                         out_size=len(candidate_constraints),
                                         do_max_pool=do_max_pool,
                                         do_softmax=True)
        else:
            raise Exception("Unrecognized net_arch: {}".format(ec.net_arch))
        optimizer = optim.Adam(self._constraint_model.parameters(), lr=0.0001)
        nll_loss = nn.NLLLoss()
        itr = 0
        print("Training constraint model with {} datapoints".format(X.shape[0]), flush=True)
        while True:
            log_probs = torch.log(self._constraint_model(X))
            loss = nll_loss(log_probs, Y)
            if itr % 1000 == 0:
                print("Constraint model training loss: {}".format(loss), flush=True)
            if loss < ec.loss_thresh or itr == 50000:
                print("Constraint model training loss: {}".format(loss), flush=True)
                break
            itr += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(self._constraint_model, self._model_path)
        print("Wrote torch model out to {}".format(self._model_path), flush=True)

        # Reset solver timeout for test
        self._solver_timeout = original_solver_timeout

    def reset_test_environment(self, test_env):
        start = time.time()
        relaxed_test_env = self._relax_test_env(test_env)
        if relaxed_test_env is None:
            self._test_policy = "random"
            return time.time()-start
        relaxed_policy = self._solver.solve(
            relaxed_test_env, timeout=self._solver_timeout)
        self._test_policy = self._get_wrapped_policy(
            relaxed_policy, relaxed_test_env.state_factory)
        solve_cost = time.time()-start
        return solve_cost

    def _relax_test_env(self, test_env):
        self._test_env = test_env
        if self._constraint_to_use is not None:
            constraint = self._constraint_to_use
        else:
            if self._constraint_model is None:
                self._constraint_model = torch.load(self._model_path)
            self._constraint_model.eval()
            if ec.net_arch == "FCN":
                x = torch.from_numpy(np.array(test_env.reward.features,
                                              dtype=np.float32))
            elif ec.net_arch == "CNN":
                x = torch.from_numpy(np.array(test_env.reward.features.flatten(),
                                              dtype=np.float32))
            else:
                raise Exception("Unrecognized net_arch: {}".format(ec.net_arch))
            probs = self._constraint_model(x).detach().numpy()[0]
            ind = gc.rand_state.choice(len(probs), p=probs)
            candidate_constraints = self._sorted_candidate_constraints(test_env)
            if self._train_candidate_constraints is not None:
                assert candidate_constraints == self._train_candidate_constraints
            constraint = candidate_constraints[ind]
            print("For test env {}, sampled constraint {} from model".format(
                test_env.__class__.__name__, constraint), flush=True)
            if constraint is None:
                return None
        return self._relax_env_from_constraint(test_env, constraint)

    @staticmethod
    def _sorted_candidate_constraints(env):
        csi = env.csi_structure
        candidate_constraints = list(sorted(csi.get_all_constraints()))
        return candidate_constraints

    def _score_constraint(self, constraint, env):
        # Temporarily set best constraint
        self._constraint_to_use = constraint
        print("Trying to score constraint {}, {} of {}".format(
            constraint, self._train_candidate_constraints.index(constraint),
            len(self._train_candidate_constraints)), flush=True)
        results = test_approach(env, self, render=gc.do_training_render,
                                train_or_test="train")
        mean_objective_value = results[2]
        prt = "\tMean objective value {}".format(mean_objective_value)
        if mean_objective_value > 0:
            print(colored(prt, "green"))
        else:
            print(prt)
        returns, cost = results[0], results[1]
        self._data_cache[(env.__class__.__name__, constraint)] = (returns, cost)
        with open("{}.constraintscores".format(ec.family_to_run), "wb") as f:
            pkl.dump(self._data_cache, f)
        # Undo setting best constraint
        self._constraint_to_use = None
        return mean_objective_value

    @staticmethod
    def _get_wrapped_policy(relaxed_policy, relaxed_state_factory):
        """Wrap policy to work in the original environment
        """
        def wrapped_policy(state):
            state_dict = state.todict()
            # During execution, assume we're never in limbo.
            state_dict[LIMBO] = 0
            wrapped_state = relaxed_state_factory.build(state_dict)
            return relaxed_policy(wrapped_state)
        return wrapped_policy

    def reset_episode(self):
        return 0.0

    def get_action(self, state):
        assert self._test_policy is not None, "Did you reset first?"
        start = time.time()
        if self._test_policy == "random":
            action = self._test_env.action_var.sample()
            return action, 0.0
        action = self._test_policy(state)
        step_cost = time.time()-start
        return action, step_cost

    def _relax_env_from_constraint(self, env, constraint,
                                   use_limbo=True, drop_variables=True):
        use_limbo = (use_limbo and self._use_limbo)
        drop_variables = (drop_variables and self._drop_variables)

        relevant_vars = env.csi_structure.get_relevant_variables(constraint)
        relevant_vars.discard(env.action_var)

        # Drop variables excluded by the CSI and imposed constraint
        if drop_variables:
            state_vars = tuple(relevant_vars)
        else:
            state_vars = tuple(env.state_factory.variables)

        # Add on limbo to relevant_variables.
        if use_limbo:
            state_vars = state_vars + (LIMBO,)

        ### Construct relaxed env.

        # Make state factory.
        relaxed_state_factory = StateFactory(state_vars)

        # Wrap env's transition model with relaxation that removes state vars.
        class RelaxedTransitionModel(TransitionModel):
            """Transition model of the relaxed env.
            """
            def get_state_vars(self):
                return state_vars

            def get_action_var(self, state_vars):
                return env.transition_model.get_action_var(state_vars)

            def model(self, state, action):
                orig_state = env.state_factory.build_from_partial(
                    state.todict())
                next_state_dist = env.model(orig_state, action)
                for next_state, prob in next_state_dist:
                    next_state_dict = next_state.todict()
                    if use_limbo:
                        limbo = self._get_limbo(state, next_state_dict)
                        if limbo:
                            next_state_dict[LIMBO] = 1
                        else:
                            next_state_dict[LIMBO] = 0
                    next_relaxed_state = relaxed_state_factory.build(
                        next_state_dict)
                    yield next_relaxed_state, prob

            def sample_next_state(self, state, action):
                next_state, _ = self.sample_next_state_failure(state, action)
                return next_state

            def sample_next_state_failure(self, state, action):
                orig_state = env.state_factory.build_from_partial(
                    state.todict())
                next_state, failure = env.sample_next_state_failure(
                    orig_state, action)
                next_state_dict = next_state.todict()
                if use_limbo:
                    limbo = self._get_limbo(state, next_state_dict)
                    if limbo:
                        next_state_dict[LIMBO] = 1
                    else:
                        next_state_dict[LIMBO] = 0
                next_relaxed_state = relaxed_state_factory.build(
                    next_state_dict)
                return next_relaxed_state, failure

            def ml_next_state(self, state, action):
                orig_state = env.state_factory.build_from_partial(
                    state.todict())
                next_state = env.ml_next_state(orig_state, action)
                next_state_dict = next_state.todict()
                if use_limbo:
                    limbo = self._get_limbo(state, next_state_dict)
                    if limbo:
                        next_state_dict[LIMBO] = 1
                    else:
                        next_state_dict[LIMBO] = 0
                next_relaxed_state = relaxed_state_factory.build(
                    next_state_dict)
                return next_relaxed_state

            @property
            def imposed_constraint(self):
                return constraint

            @staticmethod
            def _get_limbo(state, next_state_dict):
                """Next state is in limbo if either 1) state was already
                in limbo or 2) the constraint does not hold in the next state.
                """
                return int(state[LIMBO] or
                           not constraint.check(next_state_dict))
        relaxed_model = RelaxedTransitionModel()

        # Wrap env's reward function with limbo check.
        def relaxed_rew_fn(state, action):
            if use_limbo:
                if state[LIMBO] == 1:
                    return ac.limbo_reward, True
            return env.reward(state, action)
        relaxed_reward_fn = RewardFunction(env.reward.get_variables(),
                                           relaxed_rew_fn)

        # Make the relaxed environment class.
        class RelaxedEnv(Environment):
            """Relaxed environment class.
            """
            def __init__(self, transition_model, reward_fn, state_factory):
                super().__init__(transition_model, reward_fn, state_factory,
                                 generate_csi=False)

            def heuristic(self, state):
                return env.heuristic(state)

            def sample_initial_state(self):
                raise Exception("Should never get called!")

            def render(self, state):
                raise Exception("Should never get called!")

            @property
            def imposed_constraint(self):
                return constraint

            def get_solver_info(self, relaxation=None):
                return env.get_solver_info(relaxation=(
                    relaxed_model, relaxed_reward_fn, relaxed_state_factory))

        relaxed_env = RelaxedEnv(relaxed_model,
                                 relaxed_reward_fn,
                                 relaxed_state_factory)
        if hasattr(env, "pddlgym_env"):
            relaxed_env.pddlgym_env = env.pddlgym_env
        return relaxed_env


class ModelBasedNoDropping(ModelBased):
    """ModelBased class definition without variable dropping.
    """
    def __init__(self, solver):
        super().__init__(solver)
        self._drop_variables = False
