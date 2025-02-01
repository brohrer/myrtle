# Myrtle
A [real-time](#real-time) reinforcement learning benchmark framework that allows for
[multiple and intermittent rewards](#multiple-and-intermittent-rewards).

Myrtle connects a real-time environment with an agent via a message queue and runs them.

- [Creating Worlds](#worlds)
- [Creating Agents](#agents)

#### Heads Up
Myrtle runs on Linux only. It is not compatible
with Windows or MacOS due to the fact that it starts new
processes with `os.fork()`.
[Forking behavior is detailed here.](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)

# Getting started

## Install for off-the-shelf use

```bash
python3 -m pip install myrtle
```
or
```bash
uv add myrtle
```

## Install for editing

```bash
git clone https://codeberg.org/brohrer/myrtle.git
python3 -m pip install pip --upgrade
python3 -m pip install --editable myrtle
```
or
```bash
uv add git+https://codeberg.org/brohrer/myrtle.git
```

## Using Myrtle

To run a demo in Python

```python
import myrtle.demo
myrtle.demo.run_demo()
```
or at the command line
```bash
uv run src/myrtle/demo.py
```

To run a RL agent against a world

```python
from myrtle import bench
bench.run(AgentClass, WorldClass)
```

For example, to run a Random Single Action agent with a Stationary Multi-armed Bandit

```python
from myrtle import bench
from myrtle.agents.random_single_action import RandomSingleAction
from myrtle.worlds.stationary_bandit import StationaryBandit
bench.run(RandomSingleAction, StationaryBandit)
```

## Project layout

```text
src/
    myrtle/
        bench.py
        demo.py
        agents/
            base_agent.py
            random_single_action.py
            greedy_state_blind.py
            q_learning_eps.py
            ...
        worlds/
            base_world.py
            stationary_bandit.py
            intermittent_reward_bandit.py
            contextual_bandit.py
            ...
tests/
    README.md
    test_base_agent.py
    test_base_world.py
    test_bench.py
    ...
```

The `run()` function in `bench.py` is the entry point.

Run the unit test suite with `pytest`. These typically run in less than
60 seconds.

```bash
uv run pytest
```

Run the integration tests by using pytest on a file it doesn't usually
gather from, `integration_test_suite.py`. This takes a few hours to
run.
```bash
uv run pytest -s src/myrtle/tests/integration_test_suite.py
```

# Worlds

To be compatible with the Myrtle benchmark framework, world classes have to have a few
characteristics. There is a skeleton implementation in
[`base_world.py`](base_world.py)
to use as a starting place. 

## Attributes

- `n_sensors`: `int`, a member variable with the number of sensors, the size of
the array that the world will be providing each iteration.
- `n_actions`: `int`, a member variable with the number of actions, the size of
the array that the world will be expecting each iteration.
- `n_rewards (optional)`: `int`, a member variable with
the number of rewards, the length of
the reward list that the world will be providing each iteration. If not provided,
it is assumed to be the traditional 1.
- `name`: `str`, an identifier so that the history of runs on this world can be
displayed together and compared against each other.

### Multiple and intermittent rewards

Having the possibility of more than one reward is a departure
from the typical RL problem formulation and, as far as I know,
unique to this framework. It allows for intermittent rewards, that is,
it allows for individual rewards to be missing on any given time step.
See page 10 of [this paper](https://brandonrohrer.com/cartographer) for
a bit more context

## Initialization

A World class should be initializable with a three keyword arguments, 
[multiprocessing Queues](
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue),
`sensor_q`, `action_q`, and `report_q`.

`sensor_q` is the Queue for passing messages from the world to the agent.
It provides sensor and reward information, as well as information about whether
a the current episode has terminated, or the world has ceased to exist altogether.
[More detail here.](#sensor_q-messages)

`action_q` is the Queue for passing messages from the agent to world.
It informs the world of the actions the agent has chosen to take.
[More detail here.](#action_q-messages)

`report_q` is the Queue for passing messages from the world bach to the bench process.
It helps track reward at each time step.
[More detail here.](#report_q-messages)

## Methods

Every World contains a `run()` method. This is what the bench process calls
when it forks off the world process.
It will determine how long the World runs, how many many times it starts over at
the beginning (episodes), and everything else about what is run during benchmarking:

## Real-time

A good world for benchmarking with Myrtle will be tied to a wall clock
in some way. In a perfect world, there is physical hardware involved.
But this is expensive and time consuming, so more often it is a simulation
of some sort. A good way to tie this to the wall clock is with a
pacemaker that advances the simluation step by step at a fixed cadence.
There exists such a thing in the
[`pacemaker` package](https://github.com/brohrer/pacemaker)
(`pip install pacemaker-lite`).

The sample worlds in this package all have the import line

```python
from pacemaker.pacemaker import Pacemaker
```

and use the `Pacemaker.beat()` method to keep time. 

## `BaseWorld`

There is a base implementation of a world you can use as a foundation for writing
your own. Import and extend the `BaseWorld` class.

```python
from myrtle.worlds.base_world import BaseWorld

class MyWorld(BaseWorld):
    ...
```

It takes care of the interface with the rest of the benchmarking platform,
including process management, communication, and logging.
To make it your own, override the `__init__()`, `reset()`, and `step()`
methods.

## Stock Worlds

In addition to the base world there is a very short, but growing list of sample
worlds that come with Myrtle. They are useful for developing, debugging,
and benchmarking new agents.

- Stationary Bandit  
`from myrtle.worlds.stationary_bandit import StationaryBandit`  
A multi-armed bandit where each arm has a different maximum payout and a different
expected payout.

- Non-stationary Bandit  
`from myrtle.worlds.nonstationary_bandit import NonStationaryBandit`  
A multi-armed bandit where each arm has a different maximum payout and a different
expected payout, and after a number of time steps the max and expected payouts change
for all arms.

- Intermittent-reward Bandit  
`from myrtle.worlds.intermittent_reward_bandit import IntermittentRewardBandit`  
A stationary multi-armed bandit where each arm 
reports its reward individually but with intermittent outages.

- Contextual Bandit  
`from myrtle.worlds.contextual_bandit import ContextualBandit`  
A multi-armed bandit where the order of the arms is shuffled at each time step,
but the order of the arms is reported in the sensor array.

- One Hot Contextual Bandit  
`from myrtle.worlds.one_hot_contextual_bandit import OneHotContextualBandit`  
Just like the Contextual Bandit, except that the order of the arms is
reported in a concatenation of one-hot arrays.


# Agents

An Agent class has a few defining characteristics. For an example of how
these can be implemented, check out
[`base_agent.py`](https://codeberg.org/brohrer/myrtle/src/branch/main/src/myrtle/agents/base_agent.py).

## Initialization
An Agent initializes with at least these named arguments, the same as 
described above for Worlds.

- `n_sensors`: `int`
- `n_actions`: `int`
- `n_rewards (optional)`: `int`
- `sensor_q`: [`multiprocessing.Queue`](
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
- `action_q`: [`multiprocessing.Queue`](
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)

## Other attributes
The only other attribute an Agent is expected to have is a name.

- `name`: `str`, an identifier so that the history of runs with this agent can be
displayed together and compared against each other.

## `run()` method

Every Agent contains a `run()` method. This is the method that gets
called by the bench process when it forks a new agent process.
By convention the `run()` method runs on an infinite loop, at least until it
receives the message from the World that its services are no longer needed.

## `BaseAgent`

There is a base implementation of an agent you can use as a foundation for writing
your own. Import and extend the `BaseAgent` class.

```python
from myrtle.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    ...
```

It takes care of the interface with the rest of the benchmarking platform,
including process management, communication, and logging.
To make it your own, override the `__init__()`, `reset()`, and `step()`
methods.

## Agents included

As of this writing there is a short list of agents that come with Myrtle.
These aren't intended to be very sophisticated, but they are useful for 
providing performance baselines, and they serve as examples of how to
effectively extend the `BaseAgent`.

- Random Single Action  
```from myrtle.agents.random_single_action import RandomSingleAction```  
An agent that selects one action at random each time step.

- Random Multi Action  
```from myrtle.agents.random_multi_action import RandomMultiAction```  
An agent that will randomly select one or more actions at each time step,
or none at all.

- Greedy, State-blind  
```from myrtle.agents.greedy_state_blind import GreedyStateBlind```  
An agent that will always select the action with the highest expected return.

- Greedy, State-blind, with epsilon-greedy exploration  
```from myrtle.agents.greedy_state_blind_eps import GreedyStateBlindEpsilon```  
An agent that will select the action with the highest expected return
most of the time. The rest of the time it will select a single action at random.

- Q-Learning , with epsilon-greedy exploration  
```from myrtle.agents.q_learning_eps import QLearningEpsilon```  
The classic tabular learning algorithm.
[Wikipedia](https://en.wikipedia.org/wiki/Q-learning)

- Q-Learning , with curiosity-driven exploration  
```from myrtle.agents.q_learning_curiosity import QLearningCuriosity```  
Q-Learning, but with some home-rolled curiosity-driven exploration.

## Messaging

Communication between the Agent and the World is conducted through the Queues.
Through the `sensor_q` the World passes sensor and reward information to the
Agent.  Through the `action_q` the Agent passes action commands back to the World.

The World also reports reward back to the parent bench process through a
`report_q` Queue.

### `sensor_q` messages

Roughly following the conventions of [OpenAI Gym](https://github.com/openai/gym),
messages through the `sensor_q` are dicts with one or more of the following 
key-value pairs.

- "`sensors`": `numpy.Array`, the values of all sensors.
- "`rewards`": `List`, the values of each reward. Some or all of them may be `None`.
- "`truncated`": `bool`, flag that the current episode has ended and another is being kicked off.
- "`terminated`": `bool`, flag that all episodes have ended, and no more `sensor_q`
messages will be sent.

### `action_q` messages

Messages through the `action_q` are dicts with a single key-value pair.

- "`actions`": `numpy.Array`, the values of all commanded actions.

### `report_q` messages
Messages through the `report_q` are dicts with a four key-value pairs.

- "`episode`": `int`, the count of the current episode.
- "`step`": `int`, the count of the current time step. Resets with each episode.
- "`rewards`": `List`, as described for the `sensor_q` above.
- "`terminated`": `bool`, flag that all episodes have ended, and no more `report_q`
messages will be sent.

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

## Saving and reporting results

If the bench is run with argument `record=True` (the default) then,
the total reward for every time step reported by the World is written
to a [SQLite database](https://docs.python.org/3/library/sqlite3.html),
stored locally in a database file called `bench.db`.

Reporting and visualization scripts can be written that pull from these results.


 ![Myrtle process map](/doc/myrtle_processes.png)
