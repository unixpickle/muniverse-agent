"""
anyrl environments that use the muniverse bindings.
"""

from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread

from anyrl.envs import AsyncEnv, BatchedAsyncEnv
from anyrl.envs.wrappers import BatchedFrameStack, DownsampleEnv, ObsWrapperBatcher
import gym
import muniverse


def create_env(name, num_envs, num_sub_batches, fps, max_timesteps):
    assert not num_envs % num_sub_batches, 'sub-batches must divide env count'
    return BatchedAsyncEnv([[MuniverseEnv(name, fps=fps, max_timesteps=max_timesteps)
                             for _ in range(num_envs // num_sub_batches)]
                            for _ in range(num_sub_batches)])


def wrap_env(env):
    env = ObsWrapperBatcher(env, DownsampleEnv, 4)
    env = BatchedFrameStack(env, num_images=4, concat=False)
    return env


class MuniverseEnv(AsyncEnv):
    """
    A muniverse-to-anyrl adaptor.
    """

    def __init__(self, name, fps=10, max_timesteps=3000):
        self.name = name
        self.fps = fps
        self.max_timesteps = max_timesteps

        self.spec = muniverse.spec_for_name(name)
        if self.spec is None:
            raise ValueError('unknown environment: ' + name)
        self.env = muniverse.env.Env(self.spec)

        self.action_space = self.action_converter().action_space()
        self.observation_space = gym.spaces.Box(low=0, high=0xff, dtype='uint8',
                                                shape=(self.spec['Height'], self.spec['Width'], 3))

        self._th = Thread(target=self._bg_thread)
        self._cmds = Queue()
        self._responses = Queue()
        self._th.start()

    def action_converter(self):
        """
        Create an ActionConverter to go from the gym-style
        actions to muniverse-style actions.

        Returns:
          A new ActionConverter.
        """
        if self.spec['MouseRequired']:
            if self.spec['MouseType'] == 'tap':
                return TapActions(self.spec['Width'] // 2, self.spec['Height'] // 2)
            else:
                raise ValueError('unsupported mouse type: ' + self.spec['MouseType'])
        else:
            return KeyActions(self.spec['KeyWhitelist'])

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
        converter = self.action_converter()
        timestep = 0
        while True:
            cmd = self._cmds.get()
            if cmd is None:
                return
            name = cmd[0]
            if name == 'reset':
                timestep = 0
                self.env.reset()
                converter.reset()
                self._responses.put(self.env.observe())
            elif name == 'step':
                actions = converter.actions(cmd[1])
                reward, done = self.env.step(1.0 / self.fps, *actions)
                timestep += 1
                if timestep >= self.max_timesteps:
                    done = True
                if done:
                    timestep = 0
                    self.env.reset()
                    converter.reset()
                self._responses.put((self.env.observe(), reward, done, {}))


class ActionConverter(ABC):
    """
    An abstract stateful converter from gym-style actions
    to muniverse-style actions.
    """
    @abstractmethod
    def action_space(self):
        """
        Get a gym.Space for the actions.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the state of the converter, i.e. for a new
        episode.
        """
        pass

    @abstractmethod
    def actions(self, gym_action):
        """
        Convert a gym action to a muniverse action.

        Returns:
          A sequence of objects to feed to the muniverse
            step() function as arguments.
        """
        pass


class KeyActions(ActionConverter):
    """
    An ActionConverter that maps discrete values into
    muniverse key events.
    """

    def __init__(self, key_names):
        self.keys = [muniverse.key_for_code(k) for k in key_names]
        self.pressed = [False] * len(key_names)

    def action_space(self):
        return gym.spaces.Discrete(2 ** len(self.keys))

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


class TapActions(ActionConverter):
    """
    An ActionConverter that maps discrete values into
    muniverse tap events.
    """

    def __init__(self, x, y):
        self.pressed = False
        self.act_press = muniverse.MouseAction('mousePressed', x=x, y=y, click_count=1)
        self.act_release = self.act_press.with_event('mouseReleased')

    def action_space(self):
        return gym.spaces.Discrete(2)

    def reset(self):
        self.pressed = False

    def actions(self, pressed):
        if pressed == self.pressed:
            return []
        self.pressed = pressed
        return [self.act_press if pressed else self.act_release]
