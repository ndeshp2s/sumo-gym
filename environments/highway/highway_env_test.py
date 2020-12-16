import numpy as np
import time
import gym
import environments

ENV_NAME = 'Highway-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
env.reset()

for i in range(10):
	state = env.reset()

	for j in range(1000):
		action = input('Enter action: ')
		action = int(action)
		next_state, reward, done, info = env.step(action)
		print('State: ', state[0])
		print('Action: ', action)
		print('Reward: ', reward)
		print('Next State: ', next_state[0])
		print('------------------------------------------')
		state = next_state
		time.sleep(0.1)

		if done is True:
			break