from collections import deque
import cv2
import gym
import numpy as np
from gym import spaces


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max+1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class EpisodicLiveEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLiveEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        frame = self._obs_buffer.max(axis=0)
        return frame, total_reward, done, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, h=84, w=84, grayscale=True):
        super(WarpFrame, self).__init__(env)
        self.h = h
        self.w = w
        self.grayscale = grayscale
        if grayscale:
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1]*k,)), dtype=np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return LazyFrames(list(self.frames))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return LazyFrames(list(self.frames)), reward, done, info

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def make_atari(env_id):
    env = gym.make(env_id)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = EpisodicLiveEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    return env


if __name__ == '__main__':
    env = make_atari('BreakoutNoFrameskip-v4')
    s = env.reset()
    ns, r, d, info = env.step(3)