"""TAMP solver.
"""

import re
import subprocess
import time
import os
import sys
from solvers import Solver
from structs import LIMBO
from settings import GeneralConfig as gc
from settings import ApproachConfig as ac
from settings import EnvConfig as ec
from utils import flatten


class TAMPSolver(Solver):
    """TAMP solver definition.
    """
    _stilman = False
    def _solve(self, env, timeout=None, info=None, vf=False):
        if vf:
            # TAMP solver is an online method, so just plan from the
            # initial state.
            state = env.sample_initial_state()
            orig_do_render = gc.do_render
            gc.do_render = False
            plan = self._generate_new_plan(env, state, info, timeout)
            gc.do_render = orig_do_render
            qvals = self._plan_to_qvals(env, state, plan)
            return qvals
        plan = []
        def policy(state):
            if env.reward(state, None)[1]:  # if we're already done, do nothing
                return None
            nonlocal plan
            if plan:
                # Follow current plan.
                return plan.pop(0)
            orig_do_render = gc.do_render
            gc.do_render = False
            plan = self._generate_new_plan(env, state, info, timeout)
            gc.do_render = orig_do_render
            if not plan:
                return None
            return plan.pop(0)
        return policy

    @staticmethod
    def is_online():
        return True

    def _generate_new_plan(self, env, state, info, timeout):
        """Generate a new (ground) plan.
        """
        start_time = time.time()
        get_domain_pddl = info["get_domain_pddl"]
        get_problem_pddl = info["get_problem_pddl"]
        refine_step = info["refine_step"]
        failure_to_facts = info["failure_to_facts"]
        domain_pddl = get_domain_pddl()
        problem_pddl = get_problem_pddl(state, discovered_facts=tuple())
        facts_to_task_plans = {}
        # First generate a "root" task plan.
        if self._stilman:
            assert ac.approach_names == ["FullTestPlanner"], "Only use Stilman's with FullTestPlanner"
            task_plan = ["moveclear {0} UNUSED clearpose_14_{0} "
                         "startpose_{0}".format("obj40")]
        else:
            task_plan = self._run_task_planner(
                domain_pddl, problem_pddl, timeout)
        assert task_plan is not None, "Root task can't be unsolvable..."
        facts_to_task_plans[tuple()] = task_plan
        solve_tries = 0
        while True:  # TAMP search
            solve_tries += 1
            if solve_tries > 50:
                print("\nWARNING: exceeded max solve_tries, giving up\n")
                return None
            # Pick a random task plan.
            facts, task_plan = list(facts_to_task_plans.items())[
                gc.rand_state.choice(len(facts_to_task_plans))]
            # Refine it and simulate it simultaneously.
            success, newfacts_or_plan = self._refine_and_simulate(
                task_plan, refine_step, failure_to_facts, env, state,
                start_time, timeout)
            if timeout is not None and time.time()-start_time > timeout:
                return None
            if success:
                plan = newfacts_or_plan
                print("TAMP solver generated length-{} plan: {}".format(
                    len(plan), plan))
                return plan  # success!
            if newfacts_or_plan is None:
                print("\nWARNING: exceeded max refine_tries, giving up\n")
                return None
            # Plan failed, but gave us some failure information.
            # Add a new task plan corresponding to this information.
            facts = facts+tuple(newfacts_or_plan)
            if self._stilman:
                assert len(facts) == 16
                assert len(facts_to_task_plans) == 1
                obstruction = facts[0].split()[1]
                target = facts[0].split()[2].strip("()").split("_")[-1]
                for fact in facts:
                    assert fact.split()[0].strip("()") == "obstructs"
                    assert fact.split()[1] == obstruction
                    assert fact.split()[2].strip("()").split("_")[-1] == target
                new_step = ("moveclear {0} UNUSED clearpose_14_{0} "
                            "startpose_{0}".format(obstruction))
                # Insert the new step of clearing the obstruction into the plan
                # right before the step associated with moving the target.
                for i, action in enumerate(task_plan):
                    if action.split()[1] == target:
                        task_plan.insert(i, new_step)
                        break
                else:
                    import ipdb; ipdb.set_trace()  # should never happen
                task_plan = None  # don't want to use facts_to_task_plans...
            else:
                problem_pddl = get_problem_pddl(
                    state, discovered_facts=facts)
                task_plan = self._run_task_planner(
                    domain_pddl, problem_pddl, timeout)
            if task_plan is not None:
                facts_to_task_plans[facts] = task_plan

    @staticmethod
    def _plan_to_qvals(env, init_state, plan):
        """Extract q-values from the given plan that starts from the given
        init_state. Note that the plan may be None, which we handle here.
        """
        qvals = {}
        if not plan:
            return qvals
        state = init_state
        state_seq = [state]
        for act in plan:
            state = env.sample_next_state(state, act)
            state_seq.append(state)
        for i, state in enumerate(state_seq):
            returns = sum(env.reward(state2, None)[0]*(ec.gamma**j)
                          for j, state2 in enumerate(state_seq[i:]))
            if i < len(state_seq)-1:
                qvals[tuple(flatten(state)), tuple(flatten(plan[i]))] = returns
                for j, other_act in enumerate(plan):
                    if i == j:
                        continue
                    qvals[tuple(flatten(state)), tuple(flatten(other_act))] = 0
        return qvals

    @staticmethod
    def _run_task_planner(domain_pddl, problem_pddl, timeout):
        random_number = gc.rand_state.randint(999999)
        domain_name = "domain_{}.pddl".format(random_number)
        problem_name = "problem_{}.pddl".format(random_number)
        with open(domain_name, "w") as f:
            f.write(domain_pddl)
        with open(problem_name, "w") as f:
            f.write(problem_pddl)
        cmd_str = "{} {} {} -o {} -f {}".format(
            "gtimeout" if sys.platform == "darwin" else "timeout",
            timeout if timeout is not None else 0, os.environ["FF_PATH"],
            domain_name, problem_name)
        start_time = time.time()
        output = subprocess.getoutput(cmd_str)
        end_time = time.time()
        os.remove(domain_name)
        os.remove(problem_name)
        if timeout is not None and end_time-start_time > 0.9*timeout:
            return []
        if "goal can be simplified to FALSE" in output or \
            "unsolvable" in output:
            return None  # unsolvable
        task_plan = re.findall(r"\d+?: (.+)", output.lower())
        if not task_plan and "found legal" not in output and \
           "The empty plan solves it" not in output:
            raise Exception("Plan not found with FF! Error: {}".format(output))
        # print("FF length {} plan, took {}".format(len(task_plan), end_time-start_time))
        return task_plan

    @staticmethod
    def _refine_and_simulate(task_plan, refine_step, failure_to_facts,
                             env, state, start_time, timeout):
        assert LIMBO not in state or state[LIMBO] == 0
        plan = []
        for symbolic_action in task_plan:
            refine_tries = 0
            while True:
                if timeout is not None and time.time()-start_time > timeout:
                    return False, None
                refine_tries += 1
                if refine_tries > 1000:
                    return False, None
                action = refine_step(state, symbolic_action)
                next_state, failure = env.sample_next_state_failure(
                    state, action)
                # Keep refining till we find an action that doesn't either
                # fail or put us in limbo. However, if the failure generates
                # new facts, that's useful, so return them.
                if LIMBO in next_state and next_state[LIMBO] == 1:
                    continue
                if failure is not None:
                    facts = failure_to_facts(failure)
                    if facts is not None:
                        return False, facts
                    continue
                break
            state = next_state
            plan.append(action)
        return True, plan  # success!


class TAMPSolverStilman(TAMPSolver):
    """Stilman's algorithm.
    """
    _stilman = True
