from collections import deque
from multiprocessing import Pipe, Process
import cv2
import gym
import numpy as np
from gym import spaces
SEED = 0

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

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)

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

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_id':
            remote.send(env.spec.id)
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv():
    def __init__(self, env_fns):
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.remotes[0].send(('get_id', None))
        self.env_id = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)

def make_atari(env_id, rank, episodic_life=True, clip_reward=True, frame_stack=False):
    def _fn():
        env = gym.make(env_id)
        env = NoopResetEnv(env)
        env = MaxAndSkipEnv(env)
        env.seed(SEED + rank)
        if episodic_life:
            env = EpisodicLiveEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, 4)
        return env
    return _fn


if __name__ == '__main__':
    envs = SubprocVecEnv([make_atari('BreakoutNoFrameskip-v4', r) for r in range(4)])
    s = envs.reset()
    print(s.shape)
