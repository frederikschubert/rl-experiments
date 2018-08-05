import logging
import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from environments import Environment
from algorithms.policy_iteration import PolicyIteration

LOGGER = logging.getLogger("ValueIteration")


class ValueIteration(PolicyIteration):
    def run(self):
        start_time = time.time()
        LOGGER.debug(f"Value iteration:")
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
                LOGGER.debug(f"\tMaximum value change: {round(delta, 3)}")

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
        LOGGER.debug(50 * "-")
        LOGGER.debug("Value function:")
        self.env.render_value_function(self.value_function)
        LOGGER.debug("")
        LOGGER.debug("Policy:")
        self.env.render_policy(self.policy)
        end_time = time.time()
        LOGGER.info(f'Took {int(end_time - start_time)} seconds to converge.')
