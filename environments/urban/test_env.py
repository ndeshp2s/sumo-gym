import numpy as np
import time
import gym
import environments

ENV_NAME = 'Urban-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
#env.reset()

t0 = 0.0
t1 = 50.0
n = 200
p = ((t1 - t0) / n)

env.generate_walker_trips(start_time = t0, end_time = t1, period = p)

for i in range(10):
	state = env.reset()

	for j in range(1000):
		action = 0#input('Enter action: ')
		action = int(action)
		next_state, reward, done, info = env.step(action)
		print('Done: ', done)
		# print('State: ', state[0])
		# print('Action: ', action)
		# print('Reward: ', reward)
		# print('Next State: ', next_state[0])
		print('******************************************')
		state = next_state
		time.sleep(0.1)

		if done is True:
			break

	env.generate_walker_trips(start_time = t0, end_time = t1, period = p)