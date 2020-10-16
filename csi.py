"""This file contains things related to context-specific independences (CSIs)
of an MDP.
"""

from structs import EmptyConstraint, VarConstraint, ConjunctiveConstraint, \
    DisjunctiveConstraint


class CSIStructure:
    """A class to hold the CSI structure of variables in an MDP
    (no probabilities).

    `contextual_parents` is a dict from a context/constraint to a dict
    from next-variable to (set of prev- variables / action variable),
    representing what are the parents of this next- variable in the DBN,
    if the context is satisfied. All next- variables must be represented
    in every context!
    """
    def __init__(self, contextual_parents, state_vars, action_var,
                 reward_vars):
        self._contextual_parents = contextual_parents
        self._state_vars = state_vars
        self._action_var = action_var
        self._reward_vars = reward_vars

    def get_relevant_variables(self, constraint):
        """Calculate the set of relevant_variables for a constraint.
        """
        parents = self._contextual_parents[constraint]
        # Start with the reward variables...
        relevant_variables = set(self._reward_vars)
        # then add the variables in the constraint...
        if isinstance(constraint, EmptyConstraint):
            pass  # don't need to change relevant_variables
        elif isinstance(constraint, VarConstraint):
            relevant_variables |= {constraint.variable}
        elif isinstance(constraint, (ConjunctiveConstraint,
                                     DisjunctiveConstraint)):
            relevant_variables |= constraint.all_variables()
        else:
            raise Exception("Unexpected constraint: {}".format(constraint))
        # finally, iteratively add parents of relevant variables.
        while True:
            start_len = len(relevant_variables)
            for var in relevant_variables.copy():
                if var == self._action_var:
                    continue
                relevant_variables |= set(parents[var.next])
            if len(relevant_variables) == start_len:  # we're done
                break
        # Remove variables that aren't actually in the environment. This can
        # happen if we're re-using a CSI from a more complicated version of
        # an environment.
        relevant_variables = set(filter(lambda var: (var in self._state_vars or
                                                     var == self._action_var),
                                        relevant_variables))
        return relevant_variables

    def get_all_constraints(self):
        """Get the set of constraints.
        """
        return set(self._contextual_parents)
