import os
from myrtle import config


def contains(contents, match):
    contains_match = False
    for line in contents:
        if match in line:
            contains_match = True
    return contains_match


def test_values():
    assert config.js_dir[-8:] == "monitors"
    assert type(config.monitor_host) is str
    assert type(config.mq_port) is int


def test_toml():
    with open(os.path.join(config.top_dir, "config.toml"), "rt") as f:
        config_toml = f.readlines()

    assert contains(config_toml, "monitor_host")
    assert contains(config_toml, str(config.monitor_port))
    assert contains(config_toml, "mq_port")
    assert contains(config_toml, config.mq_host)


def test_js():
    with open(os.path.join(config.js_dir, "config.js"), "rt") as f:
        config_js = f.readlines()

    assert contains(config_js, "monitor_host")
    assert contains(config_js, str(config.monitor_port))
    assert contains(config_js, "mq_port")
    assert contains(config_js, config.mq_host)
