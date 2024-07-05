# Synchronization in real-time reinforcement learning

Synchronization between agents in worlds gets trickier when the world is operating in real time, as with robots and other physical systems. In non real-time RL, the world can wait for the agent to plan its next action. It is a turn-taking scenario where the agent waits for the world to make its next move and vice versa. But for physical hardware, the world doesn't wait for the agent to compute. The instant it reports its sensor readings to the agent, they are already obsolete. The world has already moved on in its machinations.

If the world moves slowly compared to the speed at which the agent computes, this mismatch can be mostly ignored. But if computations are intensive, as with some vision applications, or compute power is limited, as with many mobile robots, then this delay can become considerable. In practical applications it can render modern algorithms infeasible. 

One workaround would be to have the world sit idle for a short period to enable agent to run itself, but this would require that timing in the world to be tuned to the needs of the agent, which is dependent on the algorithm, the hardware, and other things that the world should not have to know about.

A better way to work around this is to take advantage of the fact that robots typical have a fast internal clock, and to have them try to read a new action at every step (or few steps) of this hardware clock, rather than at every agent step. This will ensure that as soon as the new action is available, it will get incorporated into the world. 

In practice this will introduce a small delay of several hardware steps, resulting in some noise in the agent. I don’t know of any way to avoid this. This is an important feature of real time reinforcement learning that gets lost in simulations. Any real time controller or physical hardware has to deal with this delay.

One interesting case study here is the human brain moving the musculoskeletal system. The ability to predict the state of the system a short time into the future appears to be a critical component of its robust performance.



This problem is described well in [this 2019 paper](https://arxiv.org/pdf/1911.04448v4).
They propose a solution where action selection is allowed a one time step delay,
and the previous action is included in the state.
This performed well for them.


This approach has the benefit of being ignorant of the system being controlled
and the computation hardware being used to select the actions. It obtains this benefit
by making the assumption that one time step is the roughly the amount of time
needed to choose an action. If the time is much shorter than that, say
one tenth of a time step, this approach adds a gratuitous delay of 9/10 of a time step.
And in cases where action selection might take several steps,
for example where more expensive planning or slower retrieval processes are involved,
it would still fail to account for that.


https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00079/full


https://arxiv.org/pdf/2010.02966

https://scholar.google.com/scholar?cites=9537250287386467127&as_sdt=40000005&sciodt=0,22&hl=en
https://scholar.google.com/scholar?cites=13694087164023050153&as_sdt=40000005&sciodt=0,22&hl=en
