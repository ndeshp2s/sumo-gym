import os
import pickle
import cloudpickle
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.epsilon_decay import EpsilonDecay
from utils.misc import create_directory

import json

DEBUG = 0
class Trainer:
    def __init__(self, env, agent, params, exp_dir, retrain = False):
        self.env = env
        self.agent = agent
        self.params = params
        self.retrain_flag = retrain

        self.epsilon_decay = EpsilonDecay(epsilon_start = self.params.hyperparameters['epsilon_start'], epsilon_end = self.params.hyperparameters['epsilon_end'], \
                                          epsilon_decay = self.params.hyperparameters['epsilon_decay'], total_steps = self.params.hyperparameters['epsilon_steps'])

        self.epsilon = self.params.hyperparameters['epsilon_start']

        log_dir = os.path.join(exp_dir, 'logs')
        if not self.retrain_flag: create_directory(dir = log_dir)
        self.writer = SummaryWriter(log_dir = log_dir)

        # checkpoint directory
        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        if not self.retrain_flag: create_directory(dir = self.checkpoint_dir)

        # Buffer memory directory
        self.buffer_dir = os.path.join(exp_dir, 'buffer_memory')


    def train(self, pre_eps = -1, pre_steps = 0):

        self.fill_memory_buffer()
        print('------------------------------------------------------------------------------')
        print('Buffer memory size: ', self.agent.buffer.__len__())
        print('------------------------------------------------------------------------------')

        self.agent.local_network.train()
        
        total_steps = pre_steps
        ep = pre_eps

        while total_steps < self.params.training_total_steps:
            
            state = self.env.reset()

            episode_reward = 0 
            episode_steps = 0           

            for step in range(self.params.training_steps_per_episode):

                # Select action
                if DEBUG:
                    action = input('Enter action: ')
                    action = int(action)
                else:
                    action, action_values  = self.agent.pick_action(state, self.epsilon)

                # if ep%10 == 0:
                #     print('--------------------------------------------------------')
                #     print('Q-Values: ', action_values)
                #     print('--------------------------------------------------------')

                # Execute action for n times
                for i in range(0, self.params.action_repeat):
                    next_state, reward, done, info = self.env.step(action)

                # Add experience to memory of local network
                self.agent.add(state = state, action = action, reward = reward, next_state = next_state, done = done)

                # Update parameters
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # compute the loss
                loss = 0
                if self.agent.buffer.__len__() > self.params.hyperparameters['batch_size']:
                    loss = self.agent.learn(batch_size = self.params.hyperparameters['batch_size'])
                    # save the loss
                    self.writer.add_scalar('Loss per step', loss, total_steps)


                # epsilon update
                self.epsilon = self.epsilon_decay.update_linear(current_eps = self.epsilon)
                self.writer.add_scalar('Epsilon decay', self.epsilon, ep)


                if done:
                    break

            ep += 1


            # Print details of the episode
            print("-----------------------------------------------------------------------------------")
            print("Episode: %d, Reward: %5f, Loss: %4f, Epsilon: %4f, Episode_Steps: %d, , Total_Steps: %d, Info: %s" % (ep, episode_reward, loss, self.epsilon, episode_steps, total_steps, info))
            print("-----------------------------------------------------------------------------------")


            # Save episode reward, steps and loss
            self.writer.add_scalar('Reward per episode', episode_reward, ep)
            self.writer.add_scalar('Steps per episode', episode_steps, ep)

            # Save memory buffer
            if ep % self.params.hyperparameters['update_buffer_memory_frequency'] == 0:
                self.save_buffer() 

            # update training parameters
            checkpoint = {'state_dict': self.agent.local_network.state_dict(),
                            'optimizer': self.agent.optimizer.state_dict(),
                            'episode': ep,
                            'epsilon': self.epsilon,
                            'total_episodes': ep,
                            'total_steps': total_steps}
            torch.save(checkpoint, self.checkpoint_dir + '/model_and_parameters.pth')


    def retrain(self):
        # load training parameters
        checkpoint = torch.load(self.checkpoint_dir + '/model_and_parameters.pth')
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        previous_episode = checkpoint['episode']
        total_episodes = checkpoint['total_episodes']
        total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']

        self.agent.local_network.train()

        self.train(pre_eps = total_episodes, pre_steps = total_steps)


    def close(self):
        self.env.close()


    def fill_memory_buffer(self, size = 0):
        if self.retrain_flag:
            self.load_buffer()

        else:
            size = int(self.params.hyperparameters['buffer_size']/2)
            while True:
                
                state = self.env.reset()
                #self.spawner.reset(config = self.env.config, spawn_points = self.env.walker_spawn_points, ev_id = self.env.get_ego_vehicle_id())         

                for step in range(self.params.training_steps_per_episode):

                    # Select action
                    action, action_values  = self.agent.pick_action(state, self.epsilon)

                    # Execute action for n times
                    #self.spawner.run_step(step) # running spawner step

                    for i in range(0, self.params.action_repeat):
                        next_state, reward, done, info = self.env.step(action)

                    # Add experience to memory of local network
                    self.agent.add(state = state, action = action, reward = reward, next_state = next_state, done = done)

                    if done:
                        break

                print('------------------------------------------------------------------------------')
                print('Buffer memory size: ', self.agent.buffer.__len__())
                print('------------------------------------------------------------------------------')
                
                #self.spawner.close()
                #sself.env.close()

                if self.agent.buffer.__len__() >= size:
                    print('------------------------------------------------------------------------------')
                    print('BUFFER MEMORY FILLED. Now learning will start')
                    print('------------------------------------------------------------------------------')
                    break


    def save_buffer(self):
        with open(self.buffer_dir + '/replay_memory_buffer.txt', 'wb') as fp:
            cloudpickle.dump(self.agent.buffer.memory, fp)

    def load_buffer(self):
        with open(self.buffer_dir + '/replay_memory_buffer.txt', 'rb') as fp:
            self.agent.buffer.memory = cloudpickle.load(fp)
