# Myrtle
## Multiple Intermittent Reward, Real-Time Reinforcement Learning Benchmark Framework

Connects a real-time environment with an agent via a Queue and runs them.
Runs on Linux.
[Not compatible](https://docs.python.org/3/library/multiprocessing.html)
with Windows or MacOS due to the fact that it starts new
processes with `os.fork()`.

## Install for off-the-shelf use

```bash
python3 -m pip install myrtle
```

## Install for editing

```bash
git clone https://codeberg.org/brohrer/myrtle.git
```

```bash
python3 -m pip install pip --upgrade
python3 -m pip install --editable myrtle
```

## Using Myrtle

To run a RL agent against a world one time

```python
from myrtle import bench
bench.run(AgentClass, WorldClass)
```

More on [AgentClass](#agents)
and [WorldClass](#worlds).


## Project layout

```text
src/
    myrtle/
        bench.py
        agents/
            base_agent.py
            ...
        worlds/
            base_world.py
            ...
tests/
    README.md
    base_agent_test.py
    base_world_test.py
    test_test.py
    ...
```

The `run()` function in `bench.py` is the entry point.

## Testing

Run the test suite with pytest.

```bash
cd myrtle
pytest
```


## Agents

An Agent class has a few defining characteristics, implemented in
[`base_agent.py`](https://codeberg.org/brohrer/myrtle/src/branch/main/src/myrtle/agents/base_agent.py).

- Initializes with named arguments
    - `n_sensors`, `int`
    - `n_actions`, `int`
    - `n_rewards`, `int`
    - `sensor_q`, [`multiprocessing.Queue`](
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
    - `action_q`, [`multiprocessing.Queue`](
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
- Contains a `run()` method.

By convention the `run()` method runs on an infinite loop.

Communication with the agent is conducted through the Queues.
The messaf
Through `action_q`
the agent passes messages in the form of a `dict` 

Messages that can be passed to and by and agent follow the conventions
 of [OpenAI Gym](https://github.com/openai/gym). 


## Worlds

To be compatible with the Myrtle benchmark framework, world classes have to have a few
characteristics. There is a skeleton implementation in
[`base_world.py`](base_world.py)
to use as a starting place. 

### Attributes

* `n_sensors`: `int`, a member variable with the number of sensors, the size of
the array that the world will be providing each iteration.
* `n_actions`: `int`, a member variable with the number of actions, the size of
the array that the world will be expecting each iteration.
* `n_rewards (optional)`: `int`, a member variable with the number of rewards, the length of
the reward list that the world will be providing each iteration. If not provided,
it is assumed to be the traditional 1.

Having the possibility of more than one reward is a departure
from the typical RL problem formulation and, as far as I know,
unique to this framework. It allows for intermittent rewards, that is,
it allows for individual rewards to be missing on any given time step.
See page 10 of [this paper](https://brandonrohrer.com/cartographer) for
a bit more context

### Initialization

A World class should be initializable with a two keyword arguments, 
[multiprocessing Queues](
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue),
"sensor\_q" is the Queue for providing sensor and reward information,
and "action\_q" is the Queue for receiving action information.

### Real-time

A good world for benchmarking with Myrtle will be tied to a wall clock
in some way. In a perfect world, there is physical hardware involved.
But this is expensive and time consuming, so more often it is a simulation
of some sort. A good way to tie this to the wall clock is with a
pacemaker that advances the simluation by one time step at a fixed cadence.
There exists such a thing in the
[`pacemaker` package](https://github.com/brohrer/pacemaker)
(`pip install pacemaker-lite`).

The sample worlds in this package all have the import line

```python
from pacemaker.pacemaker import Pacemaker
```

and use the `Pacemaker.beat()` method to keep time. 

## Multiprocess coordination

One bit of weirdness about having the World and Agent running in separate
processes is how to handle flow control. The World will be tied to the wall clock,
advancing on a fixed cadence. It will keep providing sensor and reward
information without regard for whether the Agent is ready for it.
It will keep trying to do the next thing, regardless of whether the Agent
has had enough time to decide what action to supply. The World
does not accommodate the Agent. The responsibility for keeping up falls
entirely on the Agent.

This means that the Agent must be able to handle the case where
the World has provided multiple sensor/reward updates since
the previous iteration. It also means that the World must be prepared to have 
one, zero, or multiple action commands from the Agent.

