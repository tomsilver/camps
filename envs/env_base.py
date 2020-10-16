"""Abstract class for an Environment.
"""

import os
import pickle as pkl
import time
from collections import defaultdict
import abc
import itertools
import numpy as np
from settings import EnvConfig as ec
from structs import EmptyConstraint, VarConstraint, StateVariable, \
    ConjunctiveConstraint, DisjunctiveConstraint
from csi import CSIStructure


class Environment:
    """Represents an MDP (no internal state) and context-specific independences.
    """
    def __init__(self, transition_model, reward_fn, state_factory,
                 generate_csi=True):
        self.transition_model = transition_model
        self.reward = reward_fn
        self.state_factory = state_factory
        self.action_var = transition_model.get_action_var(
            state_factory.variables)
        assert hasattr(self.action_var, "domain")  # needed for VF baseline
        if generate_csi:
            self._generate_csi()

    def _generate_csi(self):
        if not os.path.exists("learned_csi"):
            os.mkdir("learned_csi")
        cache_file = "learned_csi/{}.cache".format(ec.family_to_run)
        # Hackery to make saving/loading contextual_parents work
        # across runs, since by default the seed used for Python
        # hashing changes across runs.
        assert "PYTHONHASHSEED" in os.environ and os.environ["PYTHONHASHSEED"] == "0", "Please run:\nexport PYTHONHASHSEED=0"
        if os.path.exists(cache_file):
            print("Loading learned contextual_parents for {}...".format(
                self.__class__.__name__))
            with open(cache_file, "rb") as f:
                contextual_parents = pkl.load(f)
        else:
            contextual_parents = self._autogenerate_contextual_parents(cache_file)
        self.csi_structure = CSIStructure(
            contextual_parents,
            self.state_factory.variables,
            self.action_var,
            self.reward.get_variables())

    @abc.abstractmethod
    def get_solver_info(self, relaxation=None):
        """Get any info that might be useful for the solver.
        Optionally takes in a relaxation, which is a tuple of:
        (relaxed trans model, relaxed reward func, relaxed state factory).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_initial_state(self):
        """Get a new initial state.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def render(self, state):
        """Render the given state.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def heuristic(self, state):
        """Heuristic used by A*.
        """
        raise NotImplementedError("Override me!")

    @property
    def imposed_constraint(self):
        """Constraint imposed on the environment
        """
        return EmptyConstraint()

    def sample_next_state(self, state, action):
        """Step the environment given a state and action.
        """
        return self.transition_model.sample_next_state(state, action)

    def ml_next_state(self, state, action):
        """Get the most likely next state given a state and action.
        """
        return self.transition_model.ml_next_state(state, action)

    def sample_next_state_failure(self, state, action):
        """Step the environment given a state and action.
        Return a tuple of next_state and reason for failure.
        """
        return self.transition_model.sample_next_state_failure(state, action)

    def model(self, state, action):
        """Get a distribution over next states given a state and action.
        """
        return self.transition_model.model(state, action)

    def get_random_state(self):
        """Used for asynchronous value iteration.
        """
        ordered_vars = list(self.transition_model.get_state_vars())
        values = [v.sample() for v in ordered_vars]
        state = self.state_factory.build(dict(zip(ordered_vars, values)))
        return state

    def get_all_states(self):
        """Used for value iteration. Beware! Might be intractable.
        """
        ordered_vars = list(self.transition_model.get_state_vars())
        all_values = list(itertools.product(*[v.domain for v in ordered_vars]))
        all_states = []
        for values in all_values:
            state = self.state_factory.build(dict(zip(ordered_vars, values)))
            all_states.append(state)
        return all_states

    @property
    def imposed_constraint(self):
        """Constraint imposed on the environment
        """
        return EmptyConstraint()

    def _autogenerate_contextual_parents(self, cache_file):
        """Return a contextual_parents data structure automatically learned
        from the transition model.
        """
        start = time.time()
        print("Learning contextual_parents for {}...".format(
            self.__class__.__name__))
        # First find parents under an empty constraint (most general context).
        print("\tLearning parents for the empty constraint")
        empty = EmptyConstraint()
        contextual_parents = {empty: self._get_approx_parents(
            empty, all_possible_parents=None)}
        with open(cache_file, "wb") as f:
            pkl.dump(contextual_parents, f)
        exog_vars = set(v.prev for v, parents
                        in contextual_parents[empty].items()
                        if self.action_var not in parents)
        # Now find parents under single-term constraints.
        print("\tLearning parents for single-term constraints")
        for var in self.state_factory.variables:
            # Ignore continuous state variables.
            if not isinstance(var, StateVariable):
                continue
            # Ignore exogenous variables.
            if var in exog_vars:
                continue
            for val in var.domain:
                constr = VarConstraint(var, (val,))
                if not constr.is_satisfiable():
                    continue
                if constr.domain_size > ec.max_constraint_domain_size:
                    continue
                if ec.family_to_run == "dinner" and \
                   (not var.name.startswith("agentin(") or val == 0):
                    # Hack constraint space.
                   continue
                all_possible_parents = {}
                for next_var, pars in contextual_parents[empty].items():
                    # For single-term constraints, only variables that were
                    # dependent under empty constraint need to be checked.
                    all_possible_parents[next_var] = pars
                contextual_parents[constr] = self._get_approx_parents(
                    constr, all_possible_parents)
                with open(cache_file, "wb") as f:
                    pkl.dump(contextual_parents, f)
        # Now consider other constraints (conjunctions & disjunctions).
        print("\tLearning parents for conjunctive & disjunctive constraints")
        valids = contextual_parents.keys()-{empty}
        for length in range(2, ec.max_constraint_length+1):
            for constr_list in itertools.combinations(valids, r=length):
                constr_list = list(sorted(set(constr_list)))
                if len(constr_list) == 1:
                    continue
                conj = ConjunctiveConstraint(constr_list)
                if ec.consider_conj_constraints and \
                   conj not in contextual_parents and \
                   conj.is_satisfiable() and \
                   conj.domain_size <= ec.max_constraint_domain_size:
                    all_possible_parents = defaultdict(set)
                    for term in constr_list:
                        for next_var, pars in contextual_parents[term].items():
                            # For conjunctive constraints, only variables that
                            # were dependent under ALL constituent constraints
                            # need to be checked.
                            all_possible_parents[next_var] &= pars
                    contextual_parents[conj] = self._get_approx_parents(
                        conj, all_possible_parents)
                    with open(cache_file, "wb") as f:
                        pkl.dump(contextual_parents, f)
                disj = DisjunctiveConstraint(constr_list)
                if ec.consider_disj_constraints and \
                   disj not in contextual_parents and \
                   disj.is_satisfiable() and \
                   disj.domain_size <= ec.max_constraint_domain_size:
                    all_possible_parents = defaultdict(set)
                    for term in constr_list:
                        for next_var, pars in contextual_parents[term].items():
                            # For disjunctive constraints, only variables that
                            # were dependent under ANY constituent constraint
                            # need to be checked.
                            all_possible_parents[next_var] |= pars
                    contextual_parents[disj] = self._get_approx_parents(
                        disj, all_possible_parents)
                    with open(cache_file, "wb") as f:
                        pkl.dump(contextual_parents, f)
        print("Done, generated {} contextual_parents in {:.5f} sec".format(
            len(contextual_parents), time.time()-start))
        print()
        return contextual_parents

    def _get_approx_parents(self, constraint, all_possible_parents=None):
        state_vars = {var for var in self.state_factory.variables
                      if not (var.name.startswith("distractor") or
                              var.name.startswith("binobj"))}
        all_parents = {var.next: set() for var in state_vars}
        for var, cur_parents in all_parents.items():
            # Let's find the parents of var.next.
            if all_possible_parents is None:
                possible_parents = set(self.state_factory.variables)|{self.action_var}
            else:
                possible_parents = all_possible_parents[var]
            if isinstance(constraint, EmptyConstraint):
                pass  # don't need to change possible_parents
            elif isinstance(constraint, VarConstraint):
                possible_parents -= {constraint.variable}
            elif isinstance(constraint, (ConjunctiveConstraint,
                                         DisjunctiveConstraint)):
                possible_parents -= constraint.all_variables()
            else:
                raise Exception("Unexpected constraint: {}".format(constraint))
            for possible_parent in possible_parents:
                if possible_parent in cur_parents:
                    continue
                if not self._is_indep_of(var, possible_parent, constraint):
                    cur_parents.add(possible_parent)
        for var in self.state_factory.variables:
            if var.name.startswith("distractor") or \
               var.name.startswith("binobj"):
                all_parents[var.next] = {var}
        return all_parents

    def _is_indep_of(self, var, possible_parent, constraint):
        """Return whether this next var is estimated to be independent
        of possible_parent, under the given constraint.
        """
        self.transition_model.trans_ind = 0  # only used by dinner domain
        for _ in range(ec.indep_check_num_states):
            # Loop over many random transitions.
            try:
                state_act = self.transition_model.get_random_constrained_transition(constraint)
            except IndexError:
                assert ec.family_to_run == "dinner"
                break  # successfully exhausted transitions
            if state_act is None:  # couldn't generate transitions for this constraint
                return False  # so infavorably assume dependence
            state, action = state_act
            model = self._marginalize(self.model(state, action), var)
            num_changes = (ec.indep_check_num_changes_action
                           if possible_parent is self.action_var
                           else ec.indep_check_num_changes_notaction)
            self.transition_model.action_ind = 0  # only used by dinner domain
            for _ in range(num_changes):
                # Try changing the value of possible_parent and see if the
                # resulting var-specific transition model is different.
                new_state, new_action = self.transition_model.update_constrained_transition(state, action, possible_parent)
                new_model = self._marginalize(self.model(new_state, new_action), var)
                if self._js_div(model, new_model) > 0.1:
                    # Transition model changed -> possible_parent affects var.
                    return False
        return True

    @staticmethod
    def _marginalize(model, var):
        marged = defaultdict(float)
        for state, prob in model:
            if isinstance(state[var.prev], (np.ndarray, list, tuple)):
                marged[tuple(state[var.prev])] += prob
            else:
                marged[state[var.prev]] += prob
        return marged

    def _js_div(self, model1, model2):
        model_mid = {}
        for key in model1.keys()|model2.keys():
            model_mid[key] = model1[key]*0.5+model2[key]*0.5
        kl1 = self._kl_div(model1, model_mid)
        kl2 = self._kl_div(model2, model_mid)
        return 0.5*kl1+0.5*kl2

    @staticmethod
    def _kl_div(model1, model2):
        return sum(model1[key]*np.log(model1[key]/model2[key])
                   for key in model1 if model1[key] > 0)


class TransitionModel:
    """Represents a transition model for an MDP.
    """
    @abc.abstractmethod
    def get_state_vars(self):
        """Get the state variables of this MDP (just for a single timestep).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_action_var(self, state_vars):
        """Get the action variable of this MDP.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_random_constrained_transition(self, constraint):
        """For learning CSIs, this function provides a heuristic that gets
        useful random transitions associated with a particular constraint.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def update_constrained_transition(self, state, action, var):
        """For learning CSIs, this function returns a new state-action pair
        that is equivalent to the given one, except with a new setting of the
        given var, which could be a state variable or the action variable.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def model(self, state, action):
        """Get a distribution over next states given a state and action.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_next_state(self, state, action):
        """Step the environment given a state and action.
        Note that this cannot just use model()
        because sometimes enumerating the next state distribution
        is intractable, while sampling is still tractable.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def ml_next_state(self, state, action):
        """Get the most likely next state given a state and action.
        """
        raise NotImplementedError("Override me!")

    def sample_next_state_failure(self, state, action):
        """Step the environment given a state and action.
        Return a tuple of next_state and reason for failure.
        This is a default implementation that can be overriden by subclasses.
        """
        return (self.sample_next_state(state, action), None)


class RewardFunction:
    """Importantly, this is only a function of certain vars.
    We need to know which; that's why this isn't just a function.
    """
    def __init__(self, variables, fn, features=None):
        self._variables = variables
        self._fn = fn
        self.features = features

    def get_variables(self):
        """Get the variables that comprise this reward function.
        """
        return self._variables

    def __call__(self, state, action):
        """Return a tuple of (reward, done) given a state.
        """
        return self._fn(state, action)
