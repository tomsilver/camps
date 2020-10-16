"""Shared structs
"""
import numpy as np
from itertools import count
import abc
import heapq as hq
from settings import GeneralConfig as gc

### Variables
class Variable:
    """Generic variable.
    """
    def __init__(self, name):
        self.name = name
        self._hash = hash(self.name)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return repr(self.name)

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    @abc.abstractmethod
    def sample(self):
        """Sample a random value.
        """
        raise NotImplementedError("Override me!")


class DiscreteVariable(Variable):
    """Represents a discrete variable. Each variable's name should be unique.
    """
    def __init__(self, name, size):
        self.size = size
        self.domain = list(range(size))
        self.arbitrary_value = self.domain[0]
        super().__init__(name)

    def sample(self):
        """Sample a random value.
        """
        return gc.rand_state.choice(self.domain)


class ContinuousVariable(Variable):
    """Represents a continuous variable.
    """
    def __init__(self, name, bounds):
        self.bounds = bounds
        self.arbitrary_value = self.bounds[0]
        super().__init__(name)

    def sample(self):
        """Sample a random value.
        """
        return gc.rand_state.uniform(low=self.bounds[0], high=self.bounds[1])


class MultiDimVariable(Variable):
    """Represents a multidimensional variable. Only necessary
       for actions."""
    def __init__(self, name, variables, domain):
        self.variables = variables
        self.domain = domain
        super().__init__(name)

    def sample(self):
        """Sample a random value.
        """
        # return [v.sample() for v in self.variables]
        return self.domain[gc.rand_state.choice(len(self.domain))]


class StateVariableBase(Variable):
    """Represents a state variable, which is a variable that has
    predecessors and successors (e.g., on-road and next-on-road).
    """
    def __init__(self, name, size, _prev_ptr=None):
        # Don't pass in a prev_ptr externally; it's only for internal use.
        super().__init__(name, size)
        if _prev_ptr is None:
            self.next = self.__class__("next-"+name, size, _prev_ptr=self)
            self.prev = None
            self.is_next = False
        else:
            self.prev = _prev_ptr
            self.next = None
            self.is_next = True


class StateVariable(StateVariableBase, DiscreteVariable):
    """Representation of a discrete state variable.
    """
    pass


class ContinuousStateVariable(StateVariableBase, ContinuousVariable):
    """Representation of a continuous state variable.
    """
    pass


LIMBO = StateVariable("limbo", 2)


### State
class State(tuple):
    """Represents assignments of the state variables.

    Parameters
    ----------
    values : tuple
    state_factory : StateFactory
    """
    def __new__(cls, values, state_factory):
        _ = state_factory
        return super(State, cls).__new__(cls, values)

    def __init__(self, values, state_factory):
        super().__init__()
        _ = values
        self.state_factory = state_factory
        self.variables = state_factory.variables

    def __contains__(self, item):
        return item in self.state_factory

    def __getitem__(self, key):
        assert isinstance(key, (DiscreteVariable, ContinuousVariable))
        idx = self.variables.index(key)
        return super().__getitem__(idx)

    def update(self, key, val):
        """Get new state with updated key's value set to val.
        """
        state_dict = self.todict()
        assert key in state_dict
        state_dict[key] = val
        return self.state_factory.build(state_dict)

    def todict(self):
        """Dict-ify this state.
        """
        return dict(zip(self.variables, self))


class StateFactory:
    """Creates states with a given set of variables.

    Parameters
    ----------
    variables : tuple(DiscreteVariable)
    """
    def __init__(self, variables):
        self.variables = variables

    def __contains__(self, item):
        if isinstance(item, str):
            return any(item == var.name for var in self.variables)
        return item in self.variables

    def build(self, assignments):
        """ Build a state given a dict of assignments.
        Parameters
        ----------
        assignments : { DiscreteVariable : value }
        """
        assert isinstance(assignments, dict)
        ordered_vals = tuple([assignments[v] for v in self.variables])
        return State(ordered_vals, self)

    def build_from_tuple(self, ordered_vals):
        """Build, but from a tuple of ordered values.
        """
        assert isinstance(ordered_vals, tuple)
        return State(ordered_vals, self)

    def build_from_partial(self, partial_assignments):
        """Build, but from a partial set of assignments. Assign ARBITRARY
        values to other variables.
        """
        assert isinstance(partial_assignments, dict)
        ordered_vals = []
        for v in self.variables:
            if v in partial_assignments:
                ordered_vals.append(partial_assignments[v])
            else:
                ordered_vals.append(v.arbitrary_value)  # arbitrary value
        return State(tuple(ordered_vals), self)


### Constraints
class Constraint:
    """Represents an abstract constraint. Can check it against assignments.
    """
    satisfiability_cache = {}

    def __init__(self):
        self._is_satisfiable = None
        self._hash = hash(str(self))

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError("Override me!")

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        return str(self) < str(other)

    def __iter__(self):
        return iter([self])

    @property
    def domain_size(self):
        """Return the total domain size of this constraint, over all its
        variables.
        """
        all_vars = self.all_variables()
        if not all_vars:
            return 0
        return np.prod([v.size for v in all_vars])

    @abc.abstractmethod
    def check(self, assignments):
        """
        Parameters
        ----------
        assignments : { Variable : any }

        Returns
        -------
        result : bool
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def all_variables(self):
        """
        Returns
        -------
        all_variables : { Variable }
            All the variables that are constrained.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def negate(self):
        """Returns a new Constraint that is the negation of this constraint.
        """
        raise NotImplementedError("Override me!")

    def sample(self):
        """Sample an assignment that satisfies this constraint.
        """
        assignments = {}
        for v in self.all_variables():
            for _ in range(gc.max_satisfy_tries):
                assignments[v] = v.sample()
                if self.check(assignments):
                    break
        assert len(assignments) == len(self.all_variables())
        return assignments

    def is_satisfiable(self):
        """Return whether ANY assignment of variables (based on their domains)
        satisfies this constraint. Useful for pruning.
        """
        if self._is_satisfiable is None:
            name = str(self)
            if name not in self.satisfiability_cache:
                self.satisfiability_cache[name] = self._check_if_satisfiable()
            self._is_satisfiable = self.satisfiability_cache[name]
        return self._is_satisfiable

    def _check_if_satisfiable(self):
        """This is almost definitely terrible. TODO.
        """
        # Search for a satisfying assignment
        all_variables = self.all_variables()

        # Try to find some assignment of the constrained vars
        counter = count()
        next_count = next(counter)
        queue = [(0, 0, next_count, {})]

        while queue:
            num_attempts, _, _, assignments = hq.heappop(queue)
            num_attempts += 1
            # Full assignment?
            # keep out of loop for empty constraint edge case
            if len(assignments) == len(all_variables):
                return True
            for v in sorted(all_variables - set(assignments.keys())):
                if isinstance(v, DiscreteVariable):
                    possible_assignments = self.get_possible_assignments(v)
                else:
                    possible_assignments = [v.sample() \
                        for _ in range(10*(1+num_attempts))]
                for assignment in possible_assignments:
                    new_assignments = assignments.copy()
                    new_assignments[v] = assignment
                    # Constraint violated
                    if not self.check(new_assignments):
                        continue
                    # Finish early
                    if len(new_assignments) == len(all_variables):
                        return True
                    next_count = next(counter)
                    hq.heappush(queue, (num_attempts, -len(new_assignments),
                                        -next_count, new_assignments))

            if next_count > gc.max_satisfy_tries:
                import ipdb; ipdb.set_trace()
                break

        return False


class EmptyConstraint(Constraint):
    """Represents an empty/trivial constraint.
    """
    def __str__(self):
        return "[]"

    def check(self, assignments):
        return True

    def all_variables(self):
        return set()

    def negate(self):
        return UniversalConstraint()


class UniversalConstraint(Constraint):
    """Represents a total constraint (always false).
    """
    def __str__(self):
        return "[false]"

    def check(self, assignments):
        return False

    def all_variables(self):
        return set()

    def negate(self):
        return EmptyConstraint()

    def is_satisfiable(self):
        return False


class VarConstraint(Constraint):
    """Represents a constraint that a given variable can only take on
    a value from allowed_values.
    """
    def __init__(self, variable, allowed_values):
        self.variable = variable
        self.allowed_values = allowed_values
        self._str = "[{} in ({})]".format(
            self.variable, ",".join([str(v) for v in self.allowed_values]))
        super().__init__()

    def __str__(self):
        return self._str

    def check(self, assignments):
        # Important for satisfiable check (see parent class)!
        # Assignments may be partial. If the variable is not
        # included in the assignment, the check passes.
        if self.variable not in assignments:
            return True
        return assignments[self.variable] in self.allowed_values

    def all_variables(self):
        return {self.variable}

    def negate(self):
        new_allowed_values = tuple([val for val in self.variable.domain \
            if val not in self.allowed_values])
        return VarConstraint(self.variable, new_allowed_values)

    def get_possible_assignments(self, variable):
        assert isinstance(variable, DiscreteVariable)
        assert variable == self.variable
        return set(self.allowed_values)


class CompoundConstraint(Constraint):  # pylint:disable=abstract-method
    """Represents a compound expression of Constraint objects.
    This class is ABSTRACT.
    """
    string_joiner = ""

    def __init__(self, constraint_list):
        # No need to consider empty constraints
        constraint_list = [c for c in constraint_list
                           if not isinstance(c, EmptyConstraint)]
        # No need to consider duplicate constraints
        constraint_list = sorted(set(constraint_list))
        self.constraint_list = constraint_list
        self._all_variables = set()
        for c in constraint_list:
            self._all_variables.update(c.all_variables())
        self._str = self.string_joiner.join(["("+str(c)+")"
                                             for c in constraint_list])
        super().__init__()

    def __str__(self):
        if not self.constraint_list:
            return "[]"
        return self._str

    def all_variables(self):
        return self._all_variables

    def __iter__(self):
        return iter(self.constraint_list)


class ConjunctiveConstraint(CompoundConstraint):
    """Represents a conjunction among Constraint objects.
    """
    string_joiner = " & "

    def check(self, assignments):
        if not self.constraint_list:
            return True
        for c in self.constraint_list:
            if not c.check(assignments):
                return False
        return True

    def negate(self):
        new_constraint_list = [c.negate() for c in self.constraint_list]
        return DisjunctiveConstraint(new_constraint_list)

    def get_possible_assignments(self, variable):
        assert isinstance(variable, DiscreteVariable)
        possible_assignments = set(variable.domain)
        for constr in self.constraint_list:
            if variable not in constr.all_variables():
                continue
            possible_assignments &= constr.get_possible_assignments(variable)
        return possible_assignments

    def _check_if_satisfiable(self):
        # First do a faster impossibility check for discrete vars
        for v in self.all_variables():
            if not isinstance(v, DiscreteVariable):
                continue
            possible_assignments = set(v.domain)
            for constr in self.constraint_list:
                if v not in constr.all_variables():
                    continue
                possible_assignments &= constr.get_possible_assignments(v)
            if len(possible_assignments) == 0:
                return False
        # Fallback to full check
        return super()._check_if_satisfiable()



class DisjunctiveConstraint(CompoundConstraint):
    """Represents a disjunction among Constraint objects.
    """
    string_joiner = " | "

    def check(self, assignments):
        if not self.constraint_list:
            return True
        for c in self.constraint_list:
            if c.check(assignments):
                return True
        return False

    def negate(self):
        new_constraint_list = [c.negate() for c in self.constraint_list]
        return ConjunctiveConstraint(new_constraint_list)

    def get_possible_assignments(self, variable):
        assert isinstance(variable, DiscreteVariable)
        possible_assignments = set(variable.domain)
        for constr in self.constraint_list:
            if variable not in constr.all_variables():
                continue
            possible_assignments |= constr.get_possible_assignments(variable)
        return possible_assignments


