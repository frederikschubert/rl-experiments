from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def simulate_step(self, state, action):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def get_states(self):
        pass

    @abstractmethod
    def render_value_function(self, value_function):
        pass

    @abstractmethod
    def render_policy(self, policy):
        pass

    @abstractmethod
    def render(self):
        pass
