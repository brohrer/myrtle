import time
from myrtle import bench
from myrtle.agents import base_agent
from myrtle.worlds import base_world
from sqlogging import logging

TIMEOUT = 10
TEST_DB_NAME = f"temp_bench_test_{int(time.time())}"


def test_result_logging():
    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        timeout=TIMEOUT,
        logging_db_name=TEST_DB_NAME,
    )
    assert exitcode == 0

    logger = logging.open_logger(name=TEST_DB_NAME)
    result = logger.query(
        f"""
        SELECT step
        FROM {TEST_DB_NAME}
        ORDER BY step_timestamp DESC
        LIMIT 1
    """
    )
    assert result[0][0] == 10

    result = logger.query(
        f"""
        SELECT AVG(reward)
        FROM {TEST_DB_NAME}
        GROUP BY episode
        ORDER BY episode DESC
        LIMIT 1
    """
    )
    assert result[0][0] > -1 and result[0][0] < 1

    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        timeout=TIMEOUT,
        db_name=TEST_DB_NAME,
    )

    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        timeout=TIMEOUT,
        db_name=TEST_DB_NAME,
    )

    result = logger.query(
        f"""
        SELECT COUNT(DISTINCT run_timestamp)
        FROM {TEST_DB_NAME}
    """
    )
    print(result)
    assert result[0][0] == 3

    logger.delete()


'''
def test_process_creation_and_destruction(setup_and_teardown):

def test_shutdown_through_sensor_q(setup_and_teardown)
    agent, world = setup_and_teardown
    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    msg = {"terminated": True}
    sen_q.put(msg)
    time.sleep(_pause)

    assert p_agent.is_alive() is False
    assert p_agent.exitcode == 0

    p_agent.close()
'''
