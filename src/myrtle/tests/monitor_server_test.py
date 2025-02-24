import requests
from threading import Thread
from myrtle.config import monitor_host, monitor_port
from myrtle.monitors import server

_pause = 0.1  # seconds


def test_server():
    Thread(target=server.serve).start()
    r = requests.get(f"http://{monitor_host}:{monitor_port}/bench.html")
    assert r.status_code == 200
    assert r.text[:15] == "<!DOCTYPE html>"
    server.shutdown()

    try:
        r = requests.get(f"http://{monitor_host}:{monitor_port}/bench.html")
        assert False
    except requests.exceptions.ConnectionError:
        assert True


if __name__ == "__main__":
    test_server()
