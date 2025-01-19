import multiprocessing as mp
import pytest
import time
import dsmq.server
from myrtle import config

_pause = 0.01
_long_pause = 0.3
_timeout = 1.0


@pytest.fixture
def setup_mq_server():
    # Kick off the dsmq server in a separate process
    p_mq = mp.Process(target=dsmq.server.serve, args=(config.MQ_HOST, config.MQ_PORT))
    p_mq.start()
    time.sleep(_long_pause)

    yield

    # Shut down the dsmq process
    p_mq.join(_timeout)

    time.sleep(_long_pause)

    if p_mq.is_alive():
        p_mq.kill()
        time.sleep(_pause)
        p_mq.close()


@pytest.fixture
def setup_mq_client():
    # Spin up a dsmq client connection
    mq = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)

    yield mq
