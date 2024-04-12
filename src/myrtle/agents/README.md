# Agents

An Agent class has a few defining characteristics, implemented in
`base_agent.py`.

- Initializes with 
[`multiprocessing.Queue`](
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
as its only argument.
- Contains a `run(queue)`
