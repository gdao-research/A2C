import os
import numpy as np
import tensorflow as tf
from model import Model
import logger
from runner import Runner
from wrappers import SubprocVecEnv, make_atari
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 0


def set_global_seed(i):
    tf.set_random_seed(i)
    np.random.seed(i)

def evaluate(env, policy):
    rewards = [0]
    for i in range(30):
        s = env.reset()
        while True:
            a = policy.get_best_action(s)
            s, r, d, info = env.step(a)
            rewards[-1] += r
            if env.env.env.env.env.was_real_done:
                rewards.append(0)
                break
            if d:
                s = env.reset()
    return rewards


if __name__ == '__main__':
    set_global_seed(113)

    with tf.Session() as sess:
        eval_env = make_atari('BreakoutNoFrameskip-v4', 113, 'eval_monitor')()
        envs = SubprocVecEnv([make_atari('BreakoutNoFrameskip-v4', r+1, './monitor') for r in range(4)])
        model = Model(eval_env.observation_space.shape, eval_env.action_space.n)
        runner = Runner(envs, model.policy)
        sess.run(tf.global_variables_initializer())

        for i in range(4000000):
            if i % 100000 == 0:
                rewards = evaluate(eval_env, model.policy)
                logger.log(f'Step: {i} | Max reward: {np.max(rewards)} | Mean reward: {np.mean(rewards):.2f} | Std: {np.std(rewards):.2f}')

            sb, ab, Rb = runner.rollout()
            pl, vl, xel, l = model.train(sb, ab, Rb)
            # if i % 1000 == 0:
            #     logger.log(f'Policy loss: {pl} | Value loss: {vl} | Cross-entropy loss: {xel} | Total loss: {l}')
