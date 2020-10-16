"""MCTS solver.
"""

import time
import numpy as np
from solvers import Solver
from settings import EnvConfig as ec
from settings import SolverConfig as sc
from settings import GeneralConfig as gc
from structs import LIMBO


class MCTS(Solver):
    """MCTS solver.
    """
    def _solve(self, env, timeout=None, info=None, vf=False):
        """Plan with MCTS. Requires only simulator access to the world.
        """
        # Use a random rollout policy.
        pi_rollout = lambda state: env.action_var.sample()
        if timeout is None:
            timeout = sc.mcts_timelimit
        else:
            timeout = min(timeout, sc.mcts_timelimit)
        if vf:
            # MCTS is an online method, so just plan from the initial state.
            state = env.sample_initial_state()
            _, all_nodes = self._mcts_build_tree(env, state, pi_rollout, timeout)
            qvals = self._mcts_get_qvals(all_nodes)
            return qvals
        def policy(state):
            root, _ = self._mcts_build_tree(env, state, pi_rollout, timeout)
            action, value = self._mcts_get_best_action(root, do_explore=False)
            if action is None:
                action, value = pi_rollout(state), 0.0
            return action
        return policy

    @staticmethod
    def is_online():
        return True

    def _mcts_build_tree(self, env, state, pi_rollout, timeout):
        root = MCTSTreeNode(state)
        for act in env.action_var.domain:
            root.children[act] = {}
        start = time.time()
        itrs = 1
        # Dict of states to tree nodes.
        all_nodes = {state: root}
        while time.time()-start < timeout:
            # Build out the tree.
            node_sequence, rewards = self._mcts_expand_tree(
                env, root, pi_rollout, all_nodes)
            itrs += 1
            for i in range(len(rewards)-2, -1, -1):  # do discounting
                rewards[i] = rewards[i]+ec.gamma*rewards[i+1]
            # Backpropagate returns up to the root.
            self._mcts_backpropagate(node_sequence, rewards)
        # print("MCTS expanded {} nodes in {} iterations".format(
        #     len(all_nodes), itrs))
        return root, all_nodes

    def _mcts_expand_tree(self, env, node, pi_rollout, all_nodes):
        step = 0
        rewards = []
        node_sequence = [node]
        while True:
            # Explore an action and take it.
            action, _ = self._mcts_get_best_action(node, do_explore=True)
            next_state = env.sample_next_state(node.state, action)
            num_tries = 0
            while True:
                if LIMBO not in next_state or next_state[LIMBO] == 0:
                    break
                action = pi_rollout(node.state)
                next_state = env.sample_next_state(node.state, action)
                num_tries += 1
                if num_tries > 50:
                    break
            rew, done = env.reward(node.state, action)
            rewards.append(rew)
            if done or step > ec.max_episode_length:
                return node_sequence, rewards
            step += 1
            if next_state not in node.children[action]:
                if next_state in all_nodes:
                    # This child already exists, reuse it.
                    child = all_nodes[next_state]
                    node.children[action][next_state] = child
                else:
                    # We discovered a new child, let's create its node.
                    child = MCTSTreeNode(next_state)
                    for act in env.action_var.domain:
                        child.children[act] = {}
                    node.children[action][next_state] = child
                    all_nodes[next_state] = child
                # Run a random policy to the end of this episode.
                assert len(node_sequence) == len(rewards)
                while True:
                    state = next_state
                    num_tries = 0
                    while True:
                        action = pi_rollout(state)
                        next_state = env.sample_next_state(state, action)
                        if LIMBO not in next_state or next_state[LIMBO] == 0:
                            break
                        num_tries += 1
                        if num_tries > 50:
                            break
                    rew, done = env.reward(state, action)
                    rewards.append(rew)
                    if done or step > ec.max_episode_length:
                        node_sequence.append(child)
                        return node_sequence, rewards
                    step += 1
            node = node.children[action][next_state]
            node_sequence.append(node)

    @staticmethod
    def _mcts_get_best_action(node, do_explore):
        best_value = float("-inf")
        best_actions = []
        for action in node.children:
            total_count = sum(node.children[action][child].count
                              for child in node.children[action])
            if total_count == 0:
                if do_explore:
                    value = float("inf")  # we want actions we haven't tried yet
                else:
                    continue
            else:
                total_reward = sum(node.children[action][child].reward
                                   for child in node.children[action])
                if do_explore:
                    bonus = sc.mcts_c*np.sqrt(
                        2*np.log(node.count)/total_count)  # UCB bonus
                else:
                    bonus = 0.0
                value = total_reward/total_count+bonus
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        if not best_actions:
            return None, None
        return gc.rand_state.choice(best_actions), best_value

    @staticmethod
    def _mcts_get_qvals(all_nodes):
        qvals = {}
        for node in all_nodes.values():
            for action in node.children:
                total_count = sum(node.children[action][child].count
                                  for child in node.children[action])
                if total_count == 0:
                    continue
                total_reward = sum(node.children[action][child].reward
                                   for child in node.children[action])
                qvals[node.state, action] = total_reward/total_count
        return qvals

    @staticmethod
    def _mcts_backpropagate(node_sequence, rewards):
        for i, node in enumerate(node_sequence):
            node.count += 1
            node.reward += rewards[i]


class MCTSTreeNode:
    """Node of a tree in MCTS.
    """
    def __init__(self, state):
        self.state = state
        self.children = {}
        self.count = 0
        self.reward = 0
