import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from environments import Environment
from algorithms.policy_iteration import PolicyIteration


class ValueIteration(PolicyIteration):
    def _run(self):
        self.logger.debug(f"Value iteration:")
        
        self._compute_value_function()
        self._compute_policy()

        self.logger.debug(50 * "-")
        self.logger.debug("Value function:")
        self.env.render_value_function(self.value_function)
        self.logger.debug("")
        self.logger.debug("Policy:")
        self.env.render_policy(self.policy)

    def _compute_value_function(self):
        '''
        Computes the value function for the optimal policy.
        '''
        while True:
            delta = 0

            for s in self.states:
                old_value = deepcopy(self.value_function[s])
                possible_actions = self.env.get_actions()
                values = []
                for a in possible_actions:
                    next_state, reward, terminal = self.env.simulate_step(s, a)
                    if terminal:
                        value = reward
                    else:
                        value = reward + self.gamma * self.value_function[next_state]
                    values.append(value)
                new_value = np.amax(values)
                self.value_function[s] = new_value
                delta = np.max([delta, np.abs(old_value - new_value)])

            if delta < self.min_delta:
                break
            else:
                self.logger.debug(f"\tMaximum value change: {round(delta, 3)}")

    def _compute_policy(self):
        '''
        Computes the optimal policy from the value function.
        '''
        for s in self.states:
            possible_actions = self.env.get_actions()
            values = []
            for a in possible_actions:
                next_state, reward, terminal = self.env.simulate_step(s, a)
                if terminal:
                    value = reward
                else:
                    value = reward + self.gamma * self.value_function[next_state]
                values.append(value)
            self.policy[s] = np.argmax(values)
