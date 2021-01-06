import numpy as np
import torch
import torch.optim as optim
from neural_networks.fc import NeuralNetwork
from rl_agents.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size = 0, action_size = 0, hyperparameters = None):

        self.state_dim = state_size
        self.action_dim  = action_size
        self.hyperparameters = hyperparameters

        if self.hyperparameters["use_cuda"]:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        else:
            self.device = "cpu"

        # Initialise Q-Network
        self.local_network = NeuralNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = self.hyperparameters["learning_rate"], eps = 1e-4)
        self.criterion = torch.nn.MSELoss()

        # Initialise replay memory
        self.buffer = ReplayBuffer(buffer_size = self.hyperparameters["buffer_size"], batch_size = self.hyperparameters["batch_size"])


    def add(self, state, action, reward, next_state, done):
        self.buffer.add_experience(state = state, action = action, reward = reward, next_state = next_state, done = done)


    def learn(self, batch_size = 32, experiences = None, step = 0):
        # if experiences is None:
        #     experiences  = self.buffer.sample(batch_size)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        # ## COMPUTE THE LOSS
        # q_predicted = self.compute_predicted_q(states = states, actions = actions)

        # q_target = self.compute_target_q(next_states = next_states, rewards = rewards, dones = dones)


        batch  = self.buffer.sample(batch_size = batch_size)

        states1 = []
        states2 = []
        actions = []
        rewards = []
        next_states1 = []
        next_states2 = []

        for e in batch:                
            states1.append(e.state[0])
            states2.append(e.state[1])
            actions.append(e.action)
            rewards.append(e.reward)
            next_states1.append(e.next_state[0])
            next_states2.append(e.next_state[1])


        states1 = torch.from_numpy(np.array(states1)).float().to(self.device)
        states2 = torch.from_numpy(np.array(states2)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states1 = torch.from_numpy(np.array(next_states1)).float().to(self.device)
        next_states2 = torch.from_numpy(np.array(next_states2)).float().to(self.device)

        # Get the q values for all actions from local network
        q_predicted_all = self.local_network.forward(x1 = states1, x2 = states2)
        #Get the q value corresponding to the action executed
        q_predicted = q_predicted_all.gather(dim = 1, index = actions.unsqueeze(dim = 1)).squeeze(dim = 1)
        # Get q values for all the actions of next state
        q_next_predicted_all = self.local_network.forward(x1 = next_states1, x2 = next_states2)
        
        # get q values for the actions of next state from target netwrok
        q_next_target_all = self.local_network.forward(x1 = next_states1, x2 = next_states2)
        # get q value of action with same index as that of the action with maximum q values (from local network)
        q_next_target = q_next_target_all.gather(1, q_next_predicted_all.max(1)[1].unsqueeze(1)).squeeze(1)
        # Find target q value using Bellmann's equation
        q_target = rewards + (self.hyperparameters["discount_rate"] * q_next_target)

        # Compute the loss
        loss = self.criterion(q_predicted, q_target)

        # make previous grad zero
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # Gradient clipping
        for param in self.local_network.parameters():
            param.grad.data.clamp_(-1, 1)

        # update params
        self.optimizer.step()

        return loss.item()


    def compute_predicted_q(self, states, actions):
        # Get the q value (from local network) corresponding to the action executed
        q_predicted = self.local_network(states).gather(1, actions.long())
        return q_predicted


    def compute_target_q(self, next_states, rewards, dones):
        # Get the q value corrsponding to best action in next state
        q_next_predicted = self.compute_predicted_q_next(next_states = next_states)
        # Find target q value using Bellmann's equation
        q_target = rewards + (self.hyperparameters["discount_rate"] * q_next_predicted * (1 - dones))
        return q_target


    def compute_predicted_q_next(self, next_states):
        # Find the index of action (from local network) with maximum q value 
        max_action_index = self.local_network(next_states).detach().argmax(1)
        # Get the q value (from local network) corrsponding to best action in next state
        q_next_predicted = self.local_network(next_states).gather(1, max_action_index.unsqueeze(1)) 
        return q_next_predicted


    def pick_action(self, state, epsilon):
        ego_vehicle_state_tensor = torch.from_numpy(state[0]).float().unsqueeze(0).to(self.device)
        environment_state_tensor = torch.from_numpy(state[1]).float().unsqueeze(0).to(self.device)

        # Query the network
        action_values = self.local_network.forward(x1 = ego_vehicle_state_tensor, x2 = environment_state_tensor)
        #print('state_tensor:', state_tensor)

        if np.random.uniform() > epsilon:
            action = action_values.max(1)[1].item()

        else:
            action = np.random.randint(0, 4)

        return action, action_values[0].squeeze(0)