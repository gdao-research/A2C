import tensorflow as tf
import numpy as np

def set_global_seed(i):
    tf.set_random_seed(i)
    np.random.seed(i)

def create_session(parallel_threads):
    config = tf.ConfigProto(intra_op_parallelism_threads=parallel_threads, inter_op_parallelism_threads=parallel_threads)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def noisy_action(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), axis=-1)

def entropy(logits, name='entropy'):
    with tf.name_scope(name):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0/z0
        return tf.reduce_sum(p0*(tf.log(z0) - a0), axis=-1)
