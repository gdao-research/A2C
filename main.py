import os
import time
import numpy as np
import tensorflow as tf
import logger
from config import CONFIG as C
from model import Model
from runner import Runner
from utils import create_session, set_global_seed
from wrappers import SubprocVecEnv, make_atari
set_global_seed(113)
time_stamp = time.strftime("%m-%d-%y-%H:%M:%S")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 0

def evaluate(env, policy, nb_episodes):
    rewards = [0]
    for i in range(nb_episodes):
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
    logger.configure(f'{C.env_id}/logs_{time_stamp}')
    for k, v in C._asdict().items():
        logger.record_tabular(k, v)
    logger.dump_tabular()
    max_reward = tf.placeholder(tf.float32, name='max_reward')
    mean_reward = tf.placeholder(tf.float32, name='mean_reward')
    max_summary = tf.summary.scalar('max_rew', max_reward)
    mean_summary = tf.summary.scalar('mean_rew', mean_reward)

    with create_session(0) as sess:
        eval_env = make_atari(C.env_id, 113, 'eval')()
        envs = SubprocVecEnv(
            [make_atari(C.env_id, r+1, 'train') for r in range(4)])
        model = Model(eval_env.observation_space.shape, eval_env.action_space.n)
        runner = Runner(envs, model.policy, nb_rollout=C.nb_rollout)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./{}/summary/{}'.format(C.env_id, time_stamp), sess.graph)

        for i in range(C.iterations):
            if i % C.eval_freq == 0:
                rewards = evaluate(eval_env, model.policy, C.eval_episodes)
                logger.log(f'Step: {i} | Max reward: {np.max(rewards)} | Mean reward: {np.mean(rewards):.2f} | Std: {np.std(rewards):.2f}')
                me, ma = sess.run([mean_summary, max_summary], feed_dict={mean_reward: np.mean(rewards), max_reward: np.max(rewards)})
                writer.add_summary(me, i)
                writer.add_summary(ma, i)
                writer.flush()
            sb, ab, Rb = runner.rollout()
            pl, vl, xel, l = model.train(sb, ab, Rb)
            # if i % 1000 == 0:
            #     logger.log(f'Policy loss: {pl} | Value loss: {vl} | Cross-entropy loss: {xel} | Total loss: {l}')
