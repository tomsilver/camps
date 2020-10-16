"""BFS solver with replanning. Can handle stochasticity. Assumes that the goal
is reached when the environment gives done=True.
"""

import time
import heapq
from solvers import Solver
from structs import LIMBO
from settings import EnvConfig as ec


class BFSReplan(Solver):
    """BFSReplan solver definition.
    """
    def _solve(self, env, timeout=None, info=None, vf=False):
        if vf:
            # BFSReplan solver is an online method, so just plan from the
            # initial state.
            state = env.sample_initial_state()
            plan, state_seq = self._astar(env, state, timeout)
            qvals = self._get_qvals(env, plan, state_seq)
            return qvals
        plan = []
        state_seq = []
        def policy(state):
            if env.reward(state, None)[1]:  # if we're already done, do nothing
                return None
            nonlocal plan
            nonlocal state_seq
            if plan and state == state_seq[0]:
                # State matches our expectation. Continue current plan.
                state_seq.pop(0)
                return plan.pop(0)
            # Either planning for the first time or replanning.
            plan, state_seq = self._astar(env, state, timeout)
            if plan is None:
                return None
            assert state == state_seq[0]
            state_seq.pop(0)
            return plan.pop(0)
        return policy

    @staticmethod
    def is_online():
        return True

    @staticmethod
    def _astar(env, init_state, timeout):
        pqueue = PriorityQueue()
        root = AStarNode(init_state, action_sequence=[],
                         state_sequence=[init_state], cost=0)
        pqueue.push(root, env.heuristic(init_state))
        start = time.time()
        visited = set()
        while not pqueue.is_empty():
            if timeout is not None and time.time()-start > timeout:
                return None, None
            node = pqueue.pop()
            visited.add(node.state)
            if env.reward(node.state, None)[1]:
                return node.action_sequence, node.state_sequence
            for act in env.action_var.domain:
                next_state = env.ml_next_state(node.state, act)
                if next_state in visited:
                    continue
                if LIMBO in next_state and next_state[LIMBO] == 1:
                    continue
                pqueue.push(AStarNode(next_state, node.action_sequence+[act],
                                      node.state_sequence+[next_state],
                                      node.cost+1),
                            node.cost+1+env.heuristic(next_state))
        return None, None

    @staticmethod
    def _get_qvals(env, plan, state_seq):
        qvals = {}
        if plan is None:
            return qvals
        for i, state in enumerate(state_seq):
            returns = sum(env.reward(state2, None)[0]*(ec.gamma**j)
                          for j, state2 in enumerate(state_seq[i:]))
            if i < len(state_seq)-1:
                qvals[state, plan[i]] = returns
                for other_act in env.action_var.domain:
                    if plan[i] == other_act:
                        continue
                    qvals[state, other_act] = 0
        return qvals


class BFSNode:
    """Node in the search tree for BFS.
    """
    def __init__(self, state, actions_to_here, states_to_here):
        self.state = state
        self.actions_to_here = actions_to_here
        self.states_to_here = states_to_here


class PriorityQueue:
    """Priority queue utility class.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        """Push item to the queue with given priority.
        """
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        """Remove and return lowest priority item from queue.
        """
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        """Return whether the queue is empty.
        """
        return len(self.heap) == 0

    def __len__(self):
        return len(self.heap)


class AStarNode:
    """Node in the search tree for A*.
    """
    def __init__(self, state, action_sequence, state_sequence, cost):
        self.state = state
        self.action_sequence = action_sequence
        self.state_sequence = state_sequence
        self.cost = cost
