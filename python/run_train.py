import argparse

from anyrl.algos import PPO
from anyrl.envs import BatchedAsyncEnv
from anyrl.envs.wrappers import BatchedFrameStack, DownsampleEnv, ObsWrapperBatcher
from anyrl.spaces import gym_spaces
from anyrl.utils.ppo import ppo_cli_args, ppo_kwargs, ppo_loop_kwargs, mpi_ppo_loop
from muniverse_agent import KeyboardEnv, IMPALAModel
import tensorflow as tf


def main():
    args = arg_parser().parse_args()
    print('Creating environments...')
    env = BatchedAsyncEnv([[make_env(args) for _ in range(args.num_envs)]
                           for _ in range(args.num_sub_batches)])
    env = ObsWrapperBatcher(env, DownsampleEnv, 4)
    env = BatchedFrameStack(env, num_images=4, concat=False)
    try:
        print('Creating session...')
        with tf.Session() as sess:
            print('Creating PPO graph...')
            model = IMPALAModel(sess, *gym_spaces(env))
            ppo = PPO(model, **ppo_kwargs(args))
            print('Initializing model variables...')
            sess.run(tf.global_variables_initializer())
            mpi_ppo_loop(ppo, env, **ppo_loop_kwargs(args))
    finally:
        env.close()


def make_env(args):
    return KeyboardEnv(args.env)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Knightower-v0')
    parser.add_argument('--num-envs', help='number of parallel envs per sub-batch',
                        default=16, type=int)
    parser.add_argument('--num-sub-batches', help='number of batches of envs', default=2, type=int)
    ppo_cli_args(parser)
    return parser


if __name__ == '__main__':
    main()
