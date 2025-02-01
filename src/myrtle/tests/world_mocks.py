import json
import time
import numpy as np

_pause = 0.01


def world_step_fake(mq, n_sensors, n_rewards, value=0):
    i_loop_step = int(value)
    i_episode = int(value)
    sensors = np.ones(n_sensors) * value
    rewards = np.ones(n_rewards) * value
    world_step_msg = json.dumps(
        {
            "loop_step": i_loop_step,
            "episode": i_episode,
            "sensors": sensors.tolist(),
            "rewards": rewards.tolist(),
        }
    )
    mq.put("world_step", world_step_msg)
    return value


def world_step_random(mq, n_sensors, n_rewards):
    return world_step_fake(mq, n_sensors, n_rewards, value=np.random.choice(17))


def multiepisode_world(mq, n_sensors, n_rewards):
    """
    A very brief and boring world that simulates two complete episodes.
    """
    time.sleep(_pause)
    world_step_fake(mq, n_sensors, n_rewards, 0)
    world_step_fake(mq, n_sensors, n_rewards, 1)
    mq.put("control", "truncated")
    world_step_fake(mq, n_sensors, n_rewards, 0)
    world_step_fake(mq, n_sensors, n_rewards, 1)
    mq.put("control", "terminated")
