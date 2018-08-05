import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from environments import Environment
from algorithms import Algorithm


class PolicyIteration(Algorithm):
    def __init__(self, env: Environment, min_delta: float, gamma: float):
        super().__init__()
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

    def _run(self):
        while True:
            self.iterations += 1
            self.logger.debug(f"Iteration {self.iterations}:")
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            if policy_stable:
                self.logger.debug(50 * "-")
                self.logger.debug("Value function:")
                self.env.render_value_function(self.value_function)
                self.logger.debug("")
                self.logger.debug("Policy:")
                self.env.render_policy(self.policy)
                break

    def _compute_values(self, state):
        return [self._compute_value(state, action) for action in self.env.get_actions()]

    def _compute_value(self, state, action):
        next_state, reward, terminal = self.env.simulate_step(state, action)
        if terminal:
            return reward
        else:
            return reward + self.gamma * self.value_function[next_state]

    def _policy_evaluation(self):
        """
        Updates the value function according to the current policy.
        """
        self.logger.debug(f"\tPolicy evaluation.")
        while True:
            delta = 0

            for s in self.states:
                old_value = deepcopy(self.value_function[s])
                self.value_function[s] = self._compute_value(s, self.policy[s])
                delta = np.max([delta, np.abs(old_value - self.value_function[s])])

            if delta < self.min_delta:
                break
            else:
                self.logger.debug(f"\tMaximum value change: {round(delta, 3)}")

    def _policy_improvement(self):
        """
        Updates policy according to the current value function.
        """
        self.logger.debug(f"\tPolicy improvement.")
        policy_stable = True
        for s in self.states:
            old_action = deepcopy(self.policy[s])
            values = self._compute_values(s)
            self.policy[s] = np.argmax(values)
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable
