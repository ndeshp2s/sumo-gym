import argparse
import os
import sys
import gym

from utils.load_parameters import load_parameters
import environments
from rl_agents.q_learning.dqn import DQNAgent
from experiments.trainer import Trainer
from experiments.tester import Tester



def main(args):
    if args.dir is None:
        print(' ---------- Please mention current experiment directory ----------')
        return

    # Directory of current experiment
    base_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_dir = os.path.join(base_dir, args.agent_type, args.dir)

    # load traing/testing parameters
    params = load_parameters(file = os.path.join(experiment_dir, 'params.dat'))
    # print('env: ', params.environment)
    # print('action_repeat: ', params.action_repeat)
    # print('agent: ', params.agent)
    # print('training_episodes: ', params.training_episodes)
    # print('training_steps_per_episode: ', params.training_steps_per_episode)
    # print('testing_episodes: ', params.testing_episodes)
    # print('testing_steps_per_episode: ', params.testing_steps_per_episode)
    # print('epsilon_start: ', params.hyperparameters['epsilon_start'])
    # print('epsilon_end: ', params.hyperparameters['epsilon_end'])
    # print('epsilon_decay: ', params.hyperparameters['epsilon_decay'])
    # print('epsilon_steps: ', params.hyperparameters['epsilon_steps'])
    # print('use_cuda: ', params.hyperparameters['use_cuda'])
    # print('learning_rate: ', params.hyperparameters['learning_rate'])
    # print('batch_size: ', params.hyperparameters['batch_size'])
    # print('discount_rate: ', params.hyperparameters['discount_rate'])
    # print('target_network_update_frequency: ', params.hyperparameters['target_network_update_frequency'])


     # Initialize the environment
    env = gym.make(params.environment)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    # Initialize the agent
    agent = DQNAgent(state_size = state_size, action_size = action_size, hyperparameters = params.hyperparameters)

    if args.train:
        trainer = Trainer(env = env, agent = agent, params = params, exp_dir = experiment_dir)

        try:
            trainer.train()

        except KeyboardInterrupt:
            trainer.close()
            sys.exit(0)

        finally:
            print('\ndone.')


    if args.retrain:
        trainer = Trainer(env = env, agent = agent, params = params, exp_dir = experiment_dir, retrain = True)

        try:
            trainer.retrain()

        except KeyboardInterrupt:
            trainer.close()
            sys.exit(0)

        finally:
            print('\ndone.')


    if args.test:
        tester = Tester(env = env, agent = agent, params = params, exp_dir = experiment_dir)     

        try:
            tester.test()


        except KeyboardInterrupt:
            try:
                tester.close()
                sys.exit(0)
            except SystemExit:
                tester.close()
                os._exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='re-train model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--parameters', dest='parameters', action='store_true', help='file for training/testing parameters')
    parser.add_argument('--dir', default = None, type=str, help='directory for the experiment')
    parser.add_argument('--env', default='Highway-v0', type=str, help='gym environment')
    parser.add_argument('--agent_type', default='fc_dqn', type=str, help='type of RL agent used')
    args = parser.parse_args()

    main(parser.parse_args())