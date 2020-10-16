"""An approach that learns a policy via regression at training, and just
follows it at test time.
"""

import pickle as pkl
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from approaches import Approach
from settings import ApproachConfig as ac
from settings import EnvConfig as ec
from utils import FCN, flatten, CNN


class PolicyTransfer(Approach):
    """Policy transfer class definition.
    """
    _include_theta = False

    def __init__(self, solver):
        super().__init__(solver)
        self._nn_policy = None
        self._test_policy = None
        self._train_actions = None
        self._model_path = (ac.model_path_prefix+
                            self.__class__.__name__+
                            ".model")

    def train(self, train_envs):
        X = []
        Y = []
        theta_shape = None
        self._train_actions = []
        for env in train_envs:
            print("Solving env {}".format(env.__class__.__name__), flush=True)
            assert env.reward.features is not None
            theta = np.array(env.reward.features)
            pol = self._solver.solve(env, timeout=ec.train_solver_timeout)
            traj = []
            state = env.sample_initial_state()
            num_steps = 0
            while True:
                action = pol(state)
                if action is None:
                    break
                traj.append((state, action))
                state = env.sample_next_state(state, action)
                if env.reward(state, None)[1]:  # env is done
                    break
                if num_steps == ec.max_episode_length:
                    break
                num_steps += 1
            for state, action in traj:
                if not hasattr(env.action_var, "size"):
                    state = flatten(state)
                    action = flatten(action)
                    # Only use train_actions if continuous action space.
                    self._train_actions.append(action)
                if ec.net_arch == "FCN":
                    if theta_shape is None:
                        theta_shape = theta.shape
                    else:  # all theta must be the same shape
                        assert theta_shape == theta.shape
                    X.append(np.r_[state, theta])
                elif ec.net_arch == "CNN":
                    if theta_shape is None:
                        theta_shape = theta.shape
                    else:  # all theta must be the same shape
                        assert theta_shape == theta.shape
                    X.append(np.r_[state, theta.flatten()])
                else:
                    raise Exception("Unrecognized net_arch: {}".format(
                        ec.net_arch))
                Y.append(action)
            np.save("{}.X.npy".format(self._model_path),
                    np.array(X, dtype=np.float32))
            np.save("{}.Y.npy".format(self._model_path),
                    np.array(Y, dtype=np.float32))
            with open(self._model_path+".actions", "wb") as f:
                pkl.dump(self._train_actions, f)
        # X = np.load("{}.X.npy".format(self._model_path))
        # Y = np.load("{}.Y.npy".format(self._model_path))
        # theta_shape = np.array(train_envs[0].reward.features).shape
        X = torch.from_numpy(np.array(X, dtype=np.float32))
        if hasattr(train_envs[0].action_var, "size"):
            Y = torch.from_numpy(np.array(Y, dtype=np.int))
        else:
            Y = torch.from_numpy(np.array(Y, dtype=np.float32))
        if ec.net_arch == "FCN":
            if not self._include_theta:
                assert len(theta_shape) == 1
                X = X[:, :-theta_shape[0]]
                theta_shape = None  # unused
            if hasattr(train_envs[0].action_var, "size"):
                self._nn_policy = FCN(in_size=X.shape[1],
                                      hid_sizes=[100, 50, 32],
                                      out_size=train_envs[0].action_var.size,
                                      do_softmax=True)
                loss_fn = nn.NLLLoss()  # classification problem
                get_loss = lambda X, Y: loss_fn(torch.log(self._nn_policy(X)), Y)
            else:
                self._nn_policy = FCN(in_size=X.shape[1],
                                      hid_sizes=[100, 50, 32],
                                      out_size=Y.shape[1])
                loss_fn = nn.MSELoss()  # regression problem
                get_loss = lambda X, Y: loss_fn(self._nn_policy(X), Y)
        elif ec.net_arch == "CNN":
            if not self._include_theta:
                assert len(theta_shape) == 3
                theta_size = np.prod(theta_shape)
                X = X[:, :-theta_size]
                theta_shape = None  # unused
                if hasattr(train_envs[0].action_var, "size"):
                    self._nn_policy = FCN(in_size=X.shape[1],
                                          hid_sizes=[50, 32],
                                          out_size=train_envs[0].action_var.size,
                                          do_softmax=True)
                    loss_fn = nn.NLLLoss()  # classification problem
                    get_loss = lambda X, Y: loss_fn(torch.log(self._nn_policy(X)), Y)
                else:
                    self._nn_policy = FCN(in_size=X.shape[1],
                                          hid_sizes=[50, 32],
                                          out_size=Y.shape[1])
                    loss_fn = nn.MSELoss()  # regression problem
                    get_loss = lambda X, Y: loss_fn(self._nn_policy(X), Y)
            else:
                do_max_pool = (ec.family_to_run == "tampnamo")
                if hasattr(train_envs[0].action_var, "size"):
                    self._nn_policy = CNN(in_size=X.shape[1], num_channels=10,
                                          kernel_size=2, hid_sizes=[50, 32],
                                          theta_shape=theta_shape,
                                          out_size=train_envs[0].action_var.size,
                                          do_max_pool=do_max_pool,
                                          do_softmax=True)
                    loss_fn = nn.NLLLoss()  # classification problem
                    get_loss = lambda X, Y: loss_fn(torch.log(self._nn_policy(X)), Y)
                else:
                    self._nn_policy = CNN(in_size=X.shape[1], num_channels=10,
                                          kernel_size=2, hid_sizes=[50, 32],
                                          theta_shape=theta_shape,
                                          out_size=Y.shape[1],
                                          do_max_pool=do_max_pool)
                    loss_fn = nn.MSELoss()  # regression problem
                    get_loss = lambda X, Y: loss_fn(self._nn_policy(X), Y)
        else:
            raise Exception("Unrecognized net_arch: {}".format(ec.net_arch))
        optimizer = optim.Adam(self._nn_policy.parameters(), lr=0.0001)
        itr = 0
        print("Training policy w/ {} datapoints".format(X.shape[0]), flush=True)
        while True:
            loss = get_loss(X, Y)
            if itr % 1000 == 0:
                print("Policy training loss: {}".format(loss), flush=True)
            if loss < ec.loss_thresh or itr == 50000:
                print("Policy training loss: {}".format(loss), flush=True)
                break
            itr += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(self._nn_policy, self._model_path)
        print("Wrote policy out to {}".format(self._model_path), flush=True)

    def reset_test_environment(self, test_env):
        if self._nn_policy is None:
            self._nn_policy = torch.load(self._model_path)
        self._nn_policy.eval()
        if self._train_actions is None:
            with open(self._model_path+".actions", "rb") as f:
                self._train_actions = pkl.load(f)
        theta = np.array(test_env.reward.features)
        def _policy(state):
            if not hasattr(test_env.action_var, "size"):
                state = flatten(state)
            if ec.net_arch == "FCN":
                x = np.r_[state, theta]
                if not self._include_theta:
                    assert len(theta.shape) == 1
                    x = x[:-theta.shape[0]]
            elif ec.net_arch == "CNN":
                x = np.r_[state, theta.flatten()]
                if not self._include_theta:
                    assert len(theta.shape) == 3
                    theta_size = np.prod(theta.shape)
                    x = x[:-theta_size]
            else:
                raise Exception("Unrecognized net_arch: {}".format(
                    ec.net_arch))
            x = np.array(x, dtype=np.float32)
            if hasattr(test_env.action_var, "size"):
                action_probs = self._nn_policy(torch.from_numpy(x)).detach().numpy()[0]
                action = np.argmax(action_probs)
            else:
                flat_action = self._nn_policy(torch.from_numpy(x)).detach().numpy()[0]
                action = test_env.unflatten(flat_action)
            return action
        self._test_policy = _policy
        return 0.0

    def reset_episode(self):
        return 0.0

    def get_action(self, state):
        assert self._test_policy is not None, "Did you reset first?"
        start = time.time()
        action = self._test_policy(state)
        step_cost = time.time()-start
        return action, step_cost


class TaskConditionedPolicyTransfer(PolicyTransfer):
    """Task-conditioned policy transfer class definition.
    """
    _include_theta = True
