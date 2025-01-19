import multiprocessing as mp
import time
import numpy as np
from myrtle.agents.q_learning_eps import QLearningEpsilon

PAUSE = 0.01  # seconds


def test_creation():
    n_sensors = 4
    n_actions = 3
    n_rewards = 2
    sensor_q = mp.Queue()
    action_q = mp.Queue()

    agent = QLearningEpsilon(
        n_sensors=n_sensors,
        n_actions=n_actions,
        n_rewards=n_rewards,
        action_q=action_q,
        sensor_q=sensor_q,
    )
    assert agent.n_sensors == n_sensors
    assert agent.n_actions == n_actions
    assert agent.n_rewards == n_rewards


def test_sensor_q_killing():
    n_sen = 4
    n_act = 3
    n_rew = 2
    sen_q = mp.Queue()
    act_q = mp.Queue()

    agent = QLearningEpsilon(
        n_sensors=n_sen,
        n_actions=n_act,
        n_rewards=n_rew,
        action_q=act_q,
        sensor_q=sen_q,
    )
    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    msg = {"terminated": True}
    sen_q.put(msg)
    time.sleep(PAUSE)

    assert p_agent.is_alive() is False
    assert p_agent.exitcode == 0

    p_agent.close()


def test_learning_rate_updating():
    n_sen = 4
    n_act = 3
    n_rew = 2
    sen_q = mp.Queue()
    act_q = mp.Queue()

    agent = QLearningEpsilon(
        n_sensors=n_sen,
        n_actions=n_act,
        n_rewards=n_rew,
        epsilon=0.0,
        discount_factor=0.0,
        learning_rate=0.5,
        action_q=act_q,
        sensor_q=sen_q,
    )

    agent.previous_sensors = np.array([1, 2, 3, 4])
    agent.sensors = np.array([1, 2, 3, 4])
    agent.q_values[agent.previous_sensors.tobytes()] = np.zeros(agent.n_actions)
    agent.actions = np.array([0, 1, 0])
    agent.rewards = np.array([0, 128])

    agent.step()

    assert agent.q_values[agent.previous_sensors.tobytes()][0] == 0
    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 64
    assert agent.q_values[agent.previous_sensors.tobytes()][2] == 0

    agent.step()

    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 96

    agent.step()

    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 112


def test_discount_factor_updating():
    n_sen = 4
    n_act = 3
    n_rew = 2
    sen_q = mp.Queue()
    act_q = mp.Queue()

    agent = QLearningEpsilon(
        n_sensors=n_sen,
        n_actions=n_act,
        n_rewards=n_rew,
        epsilon=0.0,
        discount_factor=0.5,
        learning_rate=0.5,
        action_q=act_q,
        sensor_q=sen_q,
    )

    agent.previous_sensors = np.array([1, 2, 3, 4])
    agent.sensors = np.array([1, 2, 3, 4])
    agent.q_values[agent.previous_sensors.tobytes()] = np.zeros(agent.n_actions)
    agent.q_values[agent.sensors.tobytes()] = np.ones(agent.n_actions) * 100
    agent.actions = np.array([0, 1, 0])
    agent.rewards = np.array([0, 12])

    agent.step()

    assert agent.q_values[agent.previous_sensors.tobytes()][0] == 100
    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 81
    assert agent.q_values[agent.previous_sensors.tobytes()][2] == 100
