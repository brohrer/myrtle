# Worlds

To be compatible with the RTRL benchmark framework, world classes have to have a few
characteristics. There is
[a skeleton implementation here](
https://codeberg.org/brohrer/rtrl_bench/src/branch/main/src/worlds/base_world.py)
to use as
a starting place. 

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

A World class should be initializable with a two keyword arguments, 
[multiprocessing Queues](
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue),
"sensor\_q" is the Queue for providing sensor and reward information,
and "action\_q" is the Queue for receiving action information.
