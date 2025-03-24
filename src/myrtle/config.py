import os
import tomllib


top_dir = os.path.dirname(os.path.abspath(__file__))
js_dir = os.path.join(top_dir, "monitors")

with open(os.path.join(top_dir, "config.toml"), "rb") as f:
    _config = tomllib.load(f)


log_directory = _config["log_directory"]
monitor_host = _config["monitor_host"]
monitor_port = _config["monitor_port"]
mq_host = _config["mq_host"]
mq_port = _config["mq_port"]

monitor_frame_rate = _config["monitor_frame_rate"]


def write_config_js():
    """
        Write a .js file containing the ip addresses and ports for
        the message queue and web server. It's kludgey but it's the
        best way I've found so far of getting config parameters into
        both the Python and Javascript.

        desired format of `src/myrtle/monitors/config.js`

    export let mq_host = "192.168.1.20"
    export let mq_port = 38388
    export let monitor_host = "192.168.1.20"
    export let monitor_port = 8000
    """
    js_filename = os.path.join(js_dir, "config.js")

    # Get rid of the old version if there was one.
    try:
        os.remove(js_filename)
    except FileNotFoundError:
        pass

    with open(js_filename, "wt") as f:
        f.write(
            f"""export let mq_host = "{_config["mq_host"]}";
export let mq_port = {_config["mq_port"]};
export let monitor_host = "{_config["monitor_host"]}";
export let monitor_port = {_config["monitor_port"]};
export let monitorFrameRate = {_config["monitor_frame_rate"]};"""
        )
