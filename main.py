import logging
import numpy as np

from environments import grid_world
from algorithms import policy_iteration, value_iteration

LOGGER = logging.getLogger("main")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    env = grid_world.GridWorld(
        shape=(50, 50),
        start_location=(0, 0),
        terminal_locations=[(4, 3), (4, 4)],
        reward_locations=[(4, 3, -1.0), (4, 4, 1.0)],
    )
    algorithm1 = policy_iteration.PolicyIteration(env, min_delta=1e-15, gamma=0.9)
    algorithm1.run()
    algorithm2 = value_iteration.ValueIteration(env, min_delta=1e-15, gamma=0.9)
    algorithm2.run()

    if not np.array_equal(algorithm1.policy, algorithm2.policy):
        LOGGER.error("Algorithms did not converge to same policy!")


if __name__ == "__main__":
    main()
