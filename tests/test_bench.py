# The signal alarm functionality used here is only available in Unix.
# import signal

from myrtle import bench
from myrtle.agents import base_agent
from myrtle.worlds import base_world


# def handler(signum, frame):
#     raise RuntimeError("Timed out")

# Register the handler with a timed alarm
# signal.signal(signal.SIGALRM, handler)

def hide_test_integration():
    # Set an alarm for the allotted time in seconds
    # allotted_time = 10
    # signal.alarm(allotted_time)
    try:
        exitcode = bench.run(base_agent.BaseAgent, base_world.BaseWorld)
    except RuntimeError:
        exitcode = 1
    except Exception:
        exitcode = 1

    # signal.alarm(0)
    assert exitcode == 0
