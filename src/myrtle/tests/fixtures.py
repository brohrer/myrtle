import multiprocessing as mp
import pytest
import dsmq.server
from myrtle import config


@pytest.fixture
def setup_mq_server():
    # Kick off the dsmq server in a separate process
    p_mq_server = mp.Process(
        target=dsmq.server.serve, args=(config.MQ_HOST, config.MQ_PORT)
    )
    p_mq_server.start()

    # This client is an off switch for the server.
    # It's convoluted, but the way to cleanly shut down a dsmq server
    # running in a different process is through a command issued
    # from a client.
    mq_client = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)

    yield

    mq_client.shutdown_server()
    mq_client.close()


@pytest.fixture
def setup_mq_client():
    mq_client = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)

    yield mq_client

    mq_client.close()
