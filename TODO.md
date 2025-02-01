- get all tests passing
- write base_agent test in parallel with base_world test

- write integration test suite


- get worlds running, one by one
- initialize all dsmq.clients with host and port from config
    - takes place in BaseWorld.init_common() only
- How to handle worlds.init_common()?
    - All worlds have init_common()

- Start writing integration_test_suite.py

- get agents running one by one

- Pass all tests
- Clean out commented lines in worlds and agents

- Visual monitoring
  - For top level Myrtle
  - For world
  - For agent
  - Separate process for each
  - Parameters passed as args
  - Fixed location, size
  - Left side, top half for top level Myrtle
  - Left side, bottom half for world
  - Right side for agent

- Get Pendulum working with BucketTree, Ziptie, and FNC

- Create new Worlds
  - Cart pole
  - Double pendulum
  - Mountain car line

- Build new components
  - Incremental mean and variance estimate
