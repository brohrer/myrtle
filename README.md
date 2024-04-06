# A benchmark framework for real-time reinforcement learning

Connects a real-time environment with an agent via a Queue and runs them.
Runs on Linux.
Not compatible with Windows or MacOS due to the fact that it starts new
processes with `os.fork()`.
([details](https://docs.python.org/3/library/multiprocessing.html))

```bash
git clone https://codeberg.org/brohrer/rtrl_bench.git
```

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e rtrl_bench
```

## Usage

To run a RL agent against a world one time

```python
import rtrl_bench
rtrl_bench.run(AgentClass, WorldClass)
```

where `AgentClass` and `WorldClass` are each classes containing a `run(queue)`
method that accepts a `multiprocessing.Queue` as its only argument.
([Queue documentation](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)) 

## testing

```bash
cd rtrl_bench
pytest
```

