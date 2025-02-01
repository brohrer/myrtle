import json
from multiprocessing import Process
import pytest
import time
import numpy as np
from myrtle.worlds import base_world

# Exclude pytest fixtures from some checks because they behave in peculiar ways.
from myrtle.tests.fixtures import setup_mq_server, setup_mq_client  # noqa: F401

np.random.seed(42)
# times in seconds
_pause = 0.01
_long_pause = 0.3
_v_long_pause = 1.0
_timeout = 1.0

_n_loop_steps = 5
_n_episodes = 2
_loop_steps_per_second = 20


@pytest.fixture
def initialize_world():
    world = base_world.BaseWorld(
        n_loop_steps=_n_loop_steps,
        n_episodes=_n_episodes,
        loop_steps_per_second=_loop_steps_per_second,
    )

    yield world

    world.close()


def test_initialization(
    setup_mq_server,  # noqa: F811
    initialize_world,
):
    world = base_world.BaseWorld()
    assert world.name == "Base world"
    assert world.n_sensors == 13
    assert world.n_actions == 5
    assert world.n_rewards == 3
    assert world.n_loop_steps == 100
    assert world.n_episodes == 1
    assert world.loop_steps_per_second == pytest.approx(10.0)
    assert world.world_steps_per_second == pytest.approx(10.0)
    assert world.pm.clock_period == pytest.approx(0.10)


def test_sensor_reward_initialization(
    setup_mq_server,  # noqa: F811
    initialize_world,
):
    world = initialize_world
    world.reset()
    time.sleep(_pause)
    world.actions = np.array([0, 0, 1, 0, 0])
    world.i_loop_step = 0

    world.step_world()
    assert world.i_action == 2

    world.sense()
    assert world.sensors[1] == 0.0
    assert world.sensors[9] == -0.3
    assert world.rewards[0] == 0.2
    assert world.rewards[2] is None


def test_mq_initialization_and_close(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_world,
):
    # Also tests mq_initialization()
    world = initialize_world
    assert not hasattr(world, "mq")
    world.initialize_mq()
    assert hasattr(world, "mq")
    world.close()
    time.sleep(_pause)

    try:
        world.mq.put("test", "this should fail")
        assert False
    except Exception:
        # Intentionally casting a wide Exception net here.
        # The specific exception is (as of this writing)
        # websockets.exceptions.ConnectionClosedOK
        # but this is dependent on the internal implementation of dsmq,
        # and I don't want to expose that here.
        # The most rigorous solution is to catch that error within dsmq
        # and raise a RuntimeError instead, then catch that here.
        # But I'm not going to do that right now.
        assert True


def test_read_agent_step(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_world,
):
    mq = setup_mq_client
    world = initialize_world
    world.initialize_mq()

    mq.put("agent_step", json.dumps({"actions": [55, 66, 77, 88]}))

    time.sleep(_pause)
    world.read_agent_step()

    assert world.actions[1] == 66
    assert world.actions[3] == 88


def test_write_world_step(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_world,
):
    mq = setup_mq_client
    world = initialize_world
    world.initialize_mq()

    world.i_loop_step = 37
    world.i_episode = 111
    world.sensors = np.array([0.3, 0.0, -6.6, 0.29, 56789])
    world.rewards = [None, 0.01, None, 87]

    world.write_world_step()
    time.sleep(_pause)
    msg = json.loads(mq.get("world_step"))
    print(msg)

    assert msg["loop_step"] == 37
    assert msg["episode"] == 111
    assert msg["sensors"][2] == -6.6
    assert msg["sensors"][4] == 56789
    assert msg["rewards"][0] is None
    assert msg["rewards"][1] == 0.01


def test_sensing(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_world,
):
    world = initialize_world

    world.actions = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    world.i_loop_step = 7
    world.step_world()
    world.sense()

    assert world.sensors[3] == 1.0
    assert world.sensors[6] == -0.3
    assert world.sensors[8] == 0.5
    assert world.sensors[11] == 0.0
    assert world.rewards[0] == 0.3
    assert world.rewards[1] == -1.5
    assert world.rewards[2] == 0.375


def test_run(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_world,
):
    mq = setup_mq_client
    world = initialize_world
    world.run()

    # Get the most recent world_step message
    response = mq.get("world_step")
    while response != "":
        msg_str = response
        response = mq.get("world_step")

    world_info = json.loads(msg_str)

    assert world_info["loop_step"] == 4
    assert world_info["episode"] == 1
    assert world_info["rewards"][1] is None


def test_controlled_shutdown(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_world,
):
    mq = setup_mq_client
    world = initialize_world
    p_world = Process(target=world.run())
    p_world.start()

    mq.put("control", "shutdown")

    p_world.join(_v_long_pause)

    assert not p_world.is_alive()
