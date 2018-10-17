"""
anyrl environments that use the muniverse bindings.
"""

from anyrl.envs import AsyncEnv, BatchedAsyncEnv
from anyrl.envs.wrappers import BatchedFrameStack, DownsampleEnv, ObsWrapperBatcher
import gym
import muniverse

from queue import Queue
from threading import Thread


def create_env(name, num_envs, num_sub_batches):
    assert not num_envs % num_sub_batches, 'sub-batches must divide env count'
    return BatchedAsyncEnv([[KeyboardEnv(name) for _ in range(num_envs // num_sub_batches)]
                            for _ in range(num_sub_batches)])


def wrap_env(env):
    env = ObsWrapperBatcher(env, DownsampleEnv, 4)
    env = BatchedFrameStack(env, num_images=4, concat=False)
    return env


class KeyboardEnv(AsyncEnv):
    """
    An asynchronous environment for a keyboard-based
    muniverse environment.
    """

    def __init__(self, name, frame_time=0.1):
        self.name = name
        self.frame_time = frame_time

        self.spec = muniverse.spec_for_name(name)
        if self.spec is None:
            raise ValueError('unknown environment: ' + name)
        self.env = muniverse.env.Env(self.spec)

        self.action_space = gym.spaces.Discrete(2 ** len(self.spec['KeyWhitelist']))
        self.observation_space = gym.spaces.Box(low=0, high=0xff, dtype='uint8',
                                                shape=(self.spec['Height'], self.spec['Width'], 3))

        self._th = Thread(target=self._bg_thread)
        self._cmds = Queue()
        self._responses = Queue()
        self._th.start()

    def close(self):
        self.env.close()
        self._cmds.put(None)
        self._th.join()

    def reset_start(self):
        self._cmds.put(('reset',))

    def reset_wait(self):
        return self._responses.get()

    def step_start(self, action):
        self._cmds.put(('step', action))

    def step_wait(self):
        return self._responses.get()

    def _bg_thread(self):
        key_actions = _KeyActions(self.spec['KeyWhitelist'])
        while True:
            cmd = self._cmds.get()
            if cmd is None:
                return
            name = cmd[0]
            if name == 'reset':
                self.env.reset()
                key_actions.reset()
                self._responses.put(self.env.observe())
            elif name == 'step':
                actions = key_actions.actions(cmd[1])
                reward, done = self.env.step(self.frame_time, *actions)
                if done:
                    self.env.reset()
                self._responses.put((self.env.observe(), reward, done, {}))


class _KeyActions:
    def __init__(self, key_names):
        self.keys = [muniverse.key_for_code(k) for k in key_names]
        self.pressed = [False] * len(key_names)

    def reset(self):
        self.pressed = [False] * len(self.pressed)

    def actions(self, pressed_bitmap):
        new_pressed = [pressed_bitmap & (2 ** i) != 0 for i in range(len(self.pressed))]
        actions = []
        for p, new_p, key in zip(self.pressed, new_pressed, self.keys):
            if p and not new_p:
                actions.append(key.with_event('keyUp'))
            elif new_p and not p:
                actions.append(key.with_event('keyDown'))
        self.pressed = new_pressed
        return actions
