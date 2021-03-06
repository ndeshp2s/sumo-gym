import os
import torch

DEBUG = 0

class Tester:
    def __init__(self, env, agent, params, exp_dir):
        self.env = env
        self.agent = agent
        self.params = params
        self.epsilon = 0

        # checkpoint directory
        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        self.load_checkpoint()


    def test(self):

        for ep in range(self.params.testing_episodes):
            state = self.env.reset()

            for step in range(self.params.testing_steps_per_episode): 
                # Select action
                if DEBUG:
                    action = input('Enter action: ')
                    action = int(action)
                else:
                    input('Enter: ')
                    action, action_values = self.agent.pick_action(state, 0)
                    print(action_values)

                # Execute action for n times
                for i in range(0, self.params.action_repeat):
                    next_state, reward, done, info = self.env.step(action = action)
                    print('reward: ', reward)
                    print('next_state: ', next_state)
                    print('speed: ', self.env.get_ev_speed())

                state = next_state

                if done:
                    break

            self.env.close()


    def load_checkpoint(self):
        # load training parameters
        checkpoint = torch.load(self.checkpoint_dir + '/model_and_parameters.pth')

        # Load network weights and biases
        self.agent.local_network.load_state_dict(checkpoint['state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])

        self.agent.local_network.eval()


    def close(self):
        self.env.close()