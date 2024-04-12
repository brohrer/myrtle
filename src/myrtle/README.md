## Project layout

```text
src/
    myrtle/
        README.md
        bench.py
        agents/
            README.md
            ...
        worlds/
            README.md
            base_world.py
            ...
tests/
    README.md
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
the previous iteration. 
