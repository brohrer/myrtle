# Myrtle
## Multiple Intermittent Reward, Real-Time Reinforcement Learning Benchmark Framework

Connects a real-time environment with an agent via a Queue and runs them.
Runs on Linux.
[Not compatible](https://docs.python.org/3/library/multiprocessing.html)
with Windows or MacOS due to the fact that it starts new
processes with `os.fork()`.

Additional documentation in [src/README.md](src/myrtle/README.md)

## Download

```bash
git clone https://codeberg.org/brohrer/myrtle.git
```

## Install

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --editable myrtle
```

## Usage

To run a RL agent against a world one time

```python
from myrtle import bench
bench.run(AgentClass, WorldClass)
```

More on [AgentClass](src/myrtle/agents/README.md)
and [WorldClass](src/myrtle/worlds/README.md).
