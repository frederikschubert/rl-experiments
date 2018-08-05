import logging
import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from environments import Environment

LOGGER = logging.getLogger('PolicyIteration')


class PolicyIteration:
    def __init__(self, env: Environment, min_delta: float, gamma: float):
        self.env = env
        self.min_delta = min_delta
        self.gamma = gamma
        self.value_function: Dict[Tuple[int, int], float] = {}
        self.policy: Dict[Tuple[int, int], int] = {}
        self.states = self.env.get_states()
        self.iterations = 0
        for s in self.states:
            self.value_function[s] = 0
            self.policy[s] = 0

    def run(self):
        start_time = time.time()
        while True:
            self.iterations += 1
            LOGGER.debug(f"Iteration {self.iterations}:")
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            if policy_stable:
                LOGGER.debug(50 * "-")
                LOGGER.debug("Value function:")
                self.env.render_value_function(self.value_function)
                LOGGER.debug("")
                LOGGER.debug("Policy:")
                self.env.render_policy(self.policy)
                break
        end_time = time.time()
        LOGGER.info(f'Took {int(end_time - start_time)} seconds to converge.')

    def _policy_evaluation(self):
        """
        Updates the value_function according to the current policy.
        """
        LOGGER.debug(f"\tPolicy evaluation.")
        while True:
            delta = 0

            for s in self.states:
                old_value = self.value_function[s]
                new_value = 0
                next_state, reward, terminal = self.env.simulate_step(s, self.policy[s])
                if terminal:
                    new_value = reward
                else:
                    new_value = reward + self.gamma * self.value_function[next_state]

                self.value_function[s] = new_value
                delta = np.max([delta, np.abs(old_value - new_value)])
            if delta < self.min_delta:
                break
            else:
                LOGGER.debug(f"\tMaximum value change: {round(delta, 3)}")

    def _policy_improvement(self):
        """
        Updates policy according to the current value_function.
        """
        LOGGER.debug(f"\tPolicy improvement.")
        policy_stable = True
        for s in self.states:
            old_action = deepcopy(self.policy[s])
            possible_actions = self.env.get_actions()
            values = []
            for a in possible_actions:
                next_state, reward, terminal = self.env.simulate_step(s, a)
                if terminal:
                    value = reward
                else:
                    value = reward + self.gamma * self.value_function[next_state]
                values.append(value)
            new_action = np.argmax(values)
            self.policy[s] = new_action
            if old_action != new_action:
                policy_stable = False
        return policy_stable
