import logging
from copy import deepcopy
from typing import Tuple, List, Dict

import numpy as np

from environments import Environment

LOGGER = logging.getLogger("GridWorld")

RewardLocation = Tuple[int, int, float]

ACTIONS = {0: "left", 1: "down", 2: "right", 3: "up"}


class GridWorld(Environment):
    def __init__(
        self,
        shape: Tuple[int, int],
        start_location: Tuple[int, int],
        terminal_locations: List[Tuple[int, int]],
        reward_locations: List[RewardLocation],
    ):
        self.shape = shape
        self.reward_locations = reward_locations
        self.start_location = start_location
        self.terminal_locations = terminal_locations
        self.reset()

    def reset(self):
        self.state = self.start_location
        self.grid = np.zeros(self.shape, dtype=float)
        for location in self.reward_locations:
            self.grid[location[0], location[1]] = location[2]
        return self.state

    def step(self, action):
        terminal = self._is_terminal(self.state)
        if terminal:
            next_state = None
        else:
            next_state = self._get_next_location(action)

        reward = self.grid[self.state]
        self.state = next_state

        return next_state, reward, terminal

    def simulate_step(self, state, action):
        self.state = state
        next_state, reward, terminal = self.step(action)
        return next_state, reward, terminal

    def get_actions(self):
        return np.arange(0, 4)

    def get_states(self):
        states = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                states.append((x, y))
        return states

    def render_value_function(self, value_function: Dict[Tuple[int, int], float]):
        value_map = np.zeros(self.shape)
        for s in value_function.keys():
            value_map[s] = value_function[s]
        LOGGER.debug(np.array2string(value_map))

    def render_policy(self, policy: Dict[Tuple[int, int], int]):
        policy_map = np.zeros(self.shape, dtype=str)
        for s in policy.keys():
            if self._is_terminal(s):
                policy_map[s] = "X"
            else:
                policy_map[s] = ACTIONS[policy[s]]

        LOGGER.debug(np.array2string(policy_map))

    def render(self):
        grid = np.array(self.grid, dtype=str)
        grid[self.state] = "X"
        LOGGER.debug("-----------------")
        LOGGER.debug(np.array2string(grid))
        LOGGER.debug("-----------------")

    def _is_terminal(self, location):
        for t in self.terminal_locations:
            if t[0] == location[0] and t[1] == location[1]:
                return True
        return False

    def _get_next_location(self, action):
        old_location = deepcopy(self.state)
        if action == 0:
            # up
            new_location = (self.state[0], self.state[1] - 1)
        elif action == 1:
            # right
            new_location = (self.state[0] + 1, self.state[1])
        elif action == 2:
            # down
            new_location = (self.state[0], self.state[1] + 1)
        else:
            # left
            new_location = (self.state[0] - 1, self.state[1])
        if self._is_off_grid(new_location):
            new_location = old_location
        return new_location

    def _is_off_grid(self, location):
        x, y = location
        return x < 0 or y < 0 or x >= self.shape[0] or y >= self.shape[1]
