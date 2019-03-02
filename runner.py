import numpy as np

class Runner(object):
    def __init__(self, env, policy, nb_rollout=5, gamma=0.99):
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()
        self.nb_rollout = nb_rollout
        self.gamma = gamma

    def rollout(self):
        # Roll
        N, H, W, C = self.obs.shape
        obs_roll = np.zeros((self.nb_rollout, N, H, W, C), dtype=self.obs.dtype)
        actions_roll = np.zeros((self.nb_rollout, N), dtype='int8')
        rewards_roll = np.zeros((self.nb_rollout, N))
        dones_roll = np.zeros((self.nb_rollout, N))
        for i in range(self.nb_rollout):
            actions = self.policy.get_actions(self.obs)
            obs_roll[i] = np.copy(self.obs)
            actions_roll[i] = np.copy(actions)
            self.obs, rewards, dones, _ = self.env.step(actions)
            rewards_roll[i] = np.copy(rewards)
            dones_roll[i] = np.copy(dones)
        values = self.policy.get_values(self.obs)

        # Re_arrange
        Rewards_batch = np.zeros((self.nb_rollout*N,))
        obs_batch = np.zeros((self.nb_rollout*N, H, W, C), dtype=self.obs.dtype)
        actions_batch = np.zeros((self.nb_rollout*N,), dtype='int8')
        rewards_batch = np.zeros((self.nb_rollout*N,))
        dones_batch = np.zeros((self.nb_rollout*N,))
        for i in range(self.obs.shape[0]):
            obs_batch[i*self.nb_rollout:(i+1)*self.nb_rollout] = obs_roll[::-1, i, :, :, :]
            actions_batch[i*self.nb_rollout:(i+1)*self.nb_rollout] = actions_roll[::-1, i]
            rewards_batch[i*self.nb_rollout:(i+1)*self.nb_rollout] = rewards_roll[::-1, i]
            dones_batch[i*self.nb_rollout:(i+1)*self.nb_rollout] = dones_roll[::-1, i]
            for j in range(self.nb_rollout):
                # V_t = r_t + gamma * V_{t+1}
                values[i] = rewards_batch[i*self.nb_rollout+j] + self.gamma*values[i]*(1.0 - dones_batch[i*self.nb_rollout+j])
                Rewards_batch[i*self.nb_rollout+j] = values[i]
        return obs_batch, actions_batch, Rewards_batch


if __name__ == '__main__':
    from wrappers import SubprocVecEnv, make_atari
    class Policy:
        def __init__(self, n):
            self.n = n

        def get_actions(self, obs):
            N = obs.shape[0]
            return np.random.randint(self.n, size=(N,))

        def get_values(self, obs):
            N = obs.shape[0]
            return np.random.randn(N)

    envs = SubprocVecEnv([make_atari('BreakoutNoFrameskip-v4', r, './monitor') for r in range(1)])
    n_actions = envs.action_space.n
    policy = Policy(n_actions)

    runner = Runner(envs, policy)
    obs, actions, Rewards = runner.rollout()
