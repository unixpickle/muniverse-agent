import argparse

from anyrl.algos import PPO
from anyrl.spaces import gym_spaces
from anyrl.utils.ppo import ppo_cli_args, ppo_kwargs, ppo_loop_kwargs, mpi_ppo_loop
from muniverse_agent import IMPALAModel, create_env, wrap_env
import tensorflow as tf


def main():
    args = arg_parser().parse_args()
    print('Creating environments...')
    env = create_env(args.env, args.num_envs, args.num_sub_batches, args.fps, args.max_timesteps)
    env = wrap_env(env)
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


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Knightower-v0')
    parser.add_argument('--num-envs', help='total number of separate environments',
                        default=32, type=int)
    parser.add_argument('--num-sub-batches', help='number of batches of environments',
                        default=2, type=int)
    parser.add_argument('--fps', help='timesteps per second', default=10, type=int)
    parser.add_argument('--max-timesteps', help='maximum timesteps per episode',
                        default=3000, type=int)
    ppo_cli_args(parser)
    return parser


if __name__ == '__main__':
    main()
