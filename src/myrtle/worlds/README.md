# Worlds

To be compatible with the Myrtle benchmark framework, world classes have to have a few
characteristics. There is a skeleton implementation in
[`base_world.py`](base_world.py)
to use as a starting place. 

## Attributes

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

## Initialization

A World class should be initializable with a two keyword arguments, 
[multiprocessing Queues](
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue),
"sensor\_q" is the Queue for providing sensor and reward information,
and "action\_q" is the Queue for receiving action information.

## Real-time

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

