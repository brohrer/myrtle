from myrtle import bench
from myrtle.agents import base_agent
from myrtle.worlds import base_world


def test_integration():
    exitcode = bench.run(base_agent.BaseAgent, base_world.BaseWorld)
    assert exitcode == 0
