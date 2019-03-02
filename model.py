import tensorflow as tf
from utils import noisy_action, entropy


class Forward(tf.keras.Model):
    def __init__(self, nb_action):
        super(Forward, self).__init__()
        self.conv2d1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.conv2d2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.conv2d3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense_logits = tf.keras.layers.Dense(nb_action, name='logits')
        self.dense_value = tf.keras.layers.Dense(1, name='value')

    def call(self, inp):
        x = tf.cast(inp, tf.float32)/255.0
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense_logits(x)
        value = self.dense_value(x)
        return logits, value

class Policy(object):
    def __init__(self, func, state_dim):
        self.state_ph = tf.placeholder(dtype=tf.uint8, shape=(None,) + tuple(state_dim), name='s')
        self.logits, self.value = func(self.state_ph)
        self.noise_action = noisy_action(self.logits)
        self.value_1d = self.value[:, 0]

    def get_actions(self, states):
        a = tf.get_default_session().run(self.noise_action, feed_dict={self.state_ph: states})
        return a

    def get_best_action(self, state):
        return tf.get_default_session().run(self.logits, feed_dict={self.state_ph: [state]}).argmax()

    def get_values(self, states):
        return tf.get_default_session().run(self.value_1d, feed_dict={self.state_ph: states})

class Model(object):
    def __init__(self, state_dim, nb_action):
        self.action_ph = tf.placeholder(dtype=tf.int32, shape=(None,), name='a')
        self.Reward_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='R')
        forward_model = Forward(nb_action)
        self.policy = Policy(forward_model, state_dim)

        adv = tf.subtract(self.Reward_ph, tf.stop_gradient(self.policy.value_1d), name='advantage')
        log_pi_a_s = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy.logits, labels=self.action_ph)
        self.policy_loss = tf.reduce_sum(adv*log_pi_a_s, name='policy_loss')
        self.value_loss = tf.nn.l2_loss(self.Reward_ph - self.policy.value_1d, name='value_loss')
        self.xentropy_loss = -tf.reduce_sum(entropy(self.policy.logits))
        self.loss = tf.truediv(self.policy_loss + 0.01*self.xentropy_loss + 0.5*self.value_loss, tf.cast(tf.shape(self.action_ph)[0], tf.float32), name='mean_loss')

        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, params))
        opt = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=1e-5)  # XXX: schedule learning_rate
        self.train_op = opt.apply_gradients(grads)

    def train(self, states, actions, Rewards):
        pl, vl, xel, l, _ = tf.get_default_session().run([self.policy_loss, self.value_loss, self.xentropy_loss, self.loss, self.train_op], feed_dict={self.policy.state_ph: states, self.action_ph: actions, self.Reward_ph: Rewards})
        return pl, vl, xel, l
