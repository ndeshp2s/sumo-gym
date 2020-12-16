import os
import sys
import time
import numpy as np
import math
from gym import Env
from gym import spaces


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

from environments.sumo_gym import SumoGym
from environments.highway import config
from utils.misc import normalize_data



class HighwayEnv(SumoGym):
    def __init__(self):
        super(HighwayEnv, self).__init__()

        self.config = config
        
        self.sumo_config = "/home/niranjan/sumo-gym/environments/highway/sumo_configs/highway.sumocfg"
        
        if self.config.use_gui:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')

        self.sumo_cmd = [self.sumo_binary, '-c', self.sumo_config, '--max-depart-delay', str(100000), '--no-step-log', '--duration-log.disable', '--random']

        # state and action space
        self.observation_space = spaces.Box(low = 0, high = 1, shape = np.array([1]), dtype = np.uint8)
        self.action_space = spaces.Discrete(config.N_DISCRETE_ACTIONS)


    def reset(self):  

        self.stop_sumo()

        time.sleep(1)

        self.start_sumo()

        self.add_ego_vehicle(self.config.ev_start_position)            # Add the ego car to the scene

        #self.initialize_ev()
        traci.simulationStep()

        observation  = self.get_observation()

        return observation


    def step(self, action):

        self.take_action(action = action)
        
        traci.simulationStep()

        observation = self.get_observation()

        reward, done, info = self.get_reward()

        return observation, reward, done, info


    def take_action(self, max_speed = 15, action = 3):
        dt = traci.simulation.getDeltaT()
        ev_speed = traci.vehicle.getSpeed(self.config.ev_id)
        desired_speed = 0.0

        # accelerate 
        if action == 0:  
            acceleration = 2.5
            desired_speed = ev_speed + dt*acceleration

        # deccelerate
        elif action == 1:
            acceleration = -2.5
            desired_speed = ev_speed + dt*acceleration
        
        # continue
        elif action == 2:
            acceleration = 0
            desired_speed = ev_speed + dt*acceleration

        # brake    
        elif action == 3:
            acceleration = -7.5
            desired_speed = ev_speed + dt*acceleration


        if desired_speed < 0.00:
            desired_speed = 0.0

        if desired_speed > 15:
            desired_speed = 15

        traci.vehicle.setSpeed(vehID = self.config.ev_id, speed = desired_speed)


    def get_observation(self):
        ego_vehicle_state = np.zeros([1])


        ego_vehicle_speed = traci.vehicle.getSpeed(self.config.ev_id)
        ego_vehicle_speed_norm = normalize_data(data = ego_vehicle_speed, min_val = 0, max_val = self.config.ev_maximum_speed)
        ego_vehicle_speed_norm = round(ego_vehicle_speed_norm, 4)
        ego_vehicle_state[0] = ego_vehicle_speed_norm

        state_tensor = []
        state_tensor.append(ego_vehicle_state)

        return state_tensor


    def get_reward(self):
        done = 0
        info = 'None'
        total_reward = d_reward = nc_reward = c_reward = 0.0

        # reward for speed
        ego_vehicle_speed = traci.vehicle.getSpeed(self.config.ev_id)
        ego_vehicle_speed = round(ego_vehicle_speed, 4)

        target_speed = self.config.ev_target_speed
        if ego_vehicle_speed > 0.0:
            d_reward = (target_speed - abs(target_speed - ego_vehicle_speed))/target_speed
        elif ego_vehicle_speed <= 0.0:
            d_reward = -1.0

        # check for goal reached
        ev_x, ev__y = traci.vehicle.getPosition(self.config.ev_id)
        dist_to_goal = math.sqrt( ( ev_x - self.config.ev_goal_position[0])**2 + ( ev__y - self.config.ev_goal_position[1])**2 )
        if dist_to_goal < 10:
            done = True
            info = 'Goal Reached'


        total_reward = d_reward
        total_reward = round(total_reward, 4)

        return total_reward, done, info



