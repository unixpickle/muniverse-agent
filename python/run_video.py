import argparse

from anyrl.envs.wrappers import BatchedWrapper
from anyrl.spaces import gym_spaces
from anyrl.utils.ffmpeg import export_video
from anyrl.utils.tf_state import load_vars
import muniverse
from muniverse_agent import IMPALAModel, create_env, wrap_env
import numpy as np
import tensorflow as tf


def main():
    args = arg_parser().parse_args()
    print('Creating environments...')
    env = create_env(args.env, 1, 1, args.fps, args.max_timesteps)
    env = ObsInInfo(env)
    env = wrap_env(env)
    try:
        print('Creating session...')
        with tf.Session() as sess:
            print('Creating model graph...')
            model = IMPALAModel(sess, *gym_spaces(env))
            print('Initializing model variables...')
            sess.run(tf.global_variables_initializer())
            load_vars(sess, args.save_path)
            print('Gathering episode...')

            def run_episode():
                states = model.start_state(1)
                env.reset_start()
                obses = env.reset_wait()
                while True:
                    outputs = model.step(obses, states)
                    env.step_start(outputs['actions'])
                    obses, _, dones, infos = env.step_wait()
                    states = outputs['states']
                    yield pad_height(infos[0]['old_obs'])
                    if dones[0]:
                        return
            spec = muniverse.spec_for_name(args.env)
            export_video(args.path, spec['Width'], padded_height(spec['Height']), 10, run_episode())
    finally:
        env.close()


def padded_height(height):
    return height + (height % 2)


def pad_height(obs):
    if obs.shape[0] % 2:
        return np.concatenate([obs, np.zeros_like(obs[0])[None]], axis=0)
    return obs


class ObsInInfo(BatchedWrapper):
    def step_wait(self, **kwargs):
        obses, rews, dones, infos = super().step_wait(**kwargs)
        for obs, info in zip(obses, infos):
            info['old_obs'] = obs
        return obses, rews, dones, infos


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to movie file', default='video.mp4')
    parser.add_argument('--env', help='environment ID', default='Knightower-v0')
    parser.add_argument('--fps', help='timesteps per second', default=10, type=int)
    parser.add_argument('--max-timesteps', help='maximum timesteps per episode',
                        default=3000, type=int)
    parser.add_argument('--save-path', help='path to trained agent', default='ppo_agent.pkl')
    return parser


if __name__ == '__main__':
    main()
