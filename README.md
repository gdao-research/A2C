# Advantage Actor Critic (A2C)

An implementation of A2C (a variant of [A3C](https://arxiv.org/pdf/1602.01783.pdf)) from OpenAI blog [post](https://blog.openai.com/baselines-acktr-a2c/).

Intuition:
  - Multiple workers work on different copies of an environment to collect a batch of data $\rightarrow$ No need for replay buffer.
  - Noise is added to logits of policy to ensure exploration.
  - Perform 1 gradient update step based on the data batch.

## Environment
- Python 3.6.5
- TensorFlow 1.12
- OpenAI Gym 0.10.5
- OpenCV 4.0.0
- mpi4py 3.0.0

\* Note: All of the environment modification were taken from OpenAI baseline [repository](https://github.com/openai/baselines/tree/master/baselines/common).
