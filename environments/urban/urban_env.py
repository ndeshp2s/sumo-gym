import os
import sys
import time
import numpy as np
from random import randint
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
import randomTrips

from environments.sumo_gym import SumoGym
from environments.urban import config
from utils.misc import normalize_data, compute_relative_position, get_index, normalize_data, euclidean_distance, ForwardVector, compute_relative_heading, ray_intersection

WALKINGAREAS = [':C_w0', ':C_w1', ':C_w2', ':C_w3', ':E_w0', ':N_w0', ':N_w1', ':N1_w0', ':S_w0', ':W_w0']
CROSSINGS = [':C_c0', ':C_c1', ':C_c2', ':C_c3', ':N2_c0']


class UrbanEnv(SumoGym):
    def __init__(self):
        super(UrbanEnv, self).__init__()

        self.config = config

        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        sumo_config = os.path.join(self.base_dir, "sumo_configs", "urban.sumocfg")
        
        if self.config.use_gui:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')

        self.sumo_cmd = [self.sumo_binary, '-c', sumo_config, '--max-depart-delay', str(100000), '--no-step-log', '--duration-log.disable', '--random']

        # state and action space
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.config.grid_height, self.config.grid_width, self.config.features), dtype = np.uint8)
        self.action_space = spaces.Discrete(config.N_DISCRETE_ACTIONS)


    def reset(self):

        self.generate_walker_trips()
        time.sleep(1)

        self.stop_sumo()
        time.sleep(1)

        self.start_sumo()

        self.add_ego_vehicle(self.config.ev_start_position)            # Add the ego car to the scene

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
            acceleration = -10.0
            desired_speed = ev_speed + dt*acceleration


        if desired_speed < 0.00:
            desired_speed = 0.0

        if desired_speed > self.config.max_speed:
            desired_speed = self.config.max_speed

        traci.vehicle.setSpeed(vehID = self.config.ev_id, speed = desired_speed)


    def get_observation(self):
        ego_vehicle_state = np.zeros([1])
        environment_state = np.zeros([self.config.grid_height, self.config.grid_width, self.config.features])

        # Get ego vehicle information
        ego_vehicle_speed = traci.vehicle.getSpeed(self.config.ev_id)
        ego_vehicle_speed_norm = normalize_data(data = ego_vehicle_speed, min_val = 0, max_val = self.config.ev_maximum_speed)
        ego_vehicle_speed_norm = round(ego_vehicle_speed_norm, 4)
        ego_vehicle_state[0] = ego_vehicle_speed_norm

        ego_vehicle_trans = self.get_transform(id = self.config.ev_id, type = 'vehicle')

        # Get walker information
        edges = traci.edge.getIDList()
        for edge in edges:
            walkers = traci.edge.getLastStepPersonIDs(edge)
            for w in walkers:
                w_trans = self.get_transform(id = w, type = 'walker')

                w_relative_position = compute_relative_position(source_transform = ego_vehicle_trans, destination_transform = w_trans)

                x_discrete, status_x = get_index(val = w_relative_position[0], start = self.config.grid_height_min, stop = self.config.grid_height_max, num = self.config.grid_height)
                y_discrete, status_y = get_index(val = w_relative_position[1], start = self.config.grid_width_min, stop = self.config.grid_width_max, num = self.config.grid_width)

                if status_x and status_y:
                    # occupancy/position
                    x_discrete = np.argmax(x_discrete)
                    y_discrete = np.argmax(y_discrete)

                    # heading
                    w_relative_heading = compute_relative_heading(source_transform = ego_vehicle_trans, destination_transform = w_trans)

                    # walker speed
                    w_speed = round(traci.person.getSpeed(w), 2)

                    # walker lane type
                    w_lane = 2 # default is side walk
                    if edge in str(CROSSINGS):
                        w_lane = 3

                    # Data normalization
                    w_relative_heading_norm = normalize_data(data = w_relative_heading, min_val = 0, max_val = 360.0)
                    w_relative_heading_norm = round(w_relative_heading_norm, 2)
                    w_speed_norm = normalize_data(data = w_speed, min_val = 0, max_val = self.config.max_speed)
                    w_speed_norm = round(w_speed_norm, 2)
                    w_lane_norm = normalize_data(data = w_lane, min_val = 0, max_val = 3)
                    w_lane_norm = round(w_lane_norm, 2)
                    w_lane_norm = round(w_lane_norm, 2)

                    # state update for walker w-> occupancy, heading, speed
                    environment_state[x_discrete, y_discrete,:] = [1.0, w_relative_heading_norm, w_speed_norm, w_lane_norm] 


        state_tensor = []
        state_tensor.append(ego_vehicle_state)
        state_tensor.append(environment_state)

        return state_tensor


    def get_reward(self):
        done = False
        info = 'None'
        total_reward = d_reward = nc_reward = c_reward = 0.0

        # reward for speed
        ego_vehicle_speed = traci.vehicle.getSpeed(self.config.ev_id)
        ego_vehicle_speed = round(ego_vehicle_speed, 2)

        target_speed = self.config.ev_target_speed
        if ego_vehicle_speed > 0.0:
            d_reward = (target_speed - abs(target_speed - ego_vehicle_speed))/target_speed
        # elif ego_vehicle_speed > self.config.target_speed:
        #     d_reward = -1.0
        elif ego_vehicle_speed <= 0.0:
            d_reward = -1.0

        # penalty for collision/near collision
        walker_list = self.get_walker_list()

        bumper_dist = 1.5
        nc_dist = (ego_vehicle_speed*ego_vehicle_speed)/(2*2.5) + bumper_dist
        nc_dist_max = max(nc_dist, (self.config.safe_distance + bumper_dist))
        nc_dist_max = round(nc_dist_max, 2)
        collision, near_collision, walker, distance = self.find_collision(walker_list = walker_list, range = nc_dist_max)

        if collision:
            if ego_vehicle_speed > 0.0:
                c_reward = -10
                done = True
                info = 'Normal Collision'
            else:
                c_reward = -10
                done = True
                info = 'Pedestrian Collision'

        elif near_collision:
            if ego_vehicle_speed > 0.0:
                print(distance/nc_dist_max)
                nc_reward = -4 * np.exp( -(distance/nc_dist_max) )
                nc_reward = round(nc_reward, 2)

        # check for goal reached
        ev_x, ev__y = traci.vehicle.getPosition(self.config.ev_id)
        dist_to_goal = math.sqrt( ( ev_x - self.config.ev_goal_position[0])**2 + ( ev__y - self.config.ev_goal_position[1])**2 )
        if dist_to_goal < 10:
            done = True
            info = 'Goal Reached'


        total_reward = d_reward + c_reward + nc_reward
        total_reward = round(total_reward, 2)
        # if collision:
        #     total_reward = c_reward
        # elif near_collision:
        #     total_reward = nc_reward
        # else:
        #     total_reward = d_reward

        total_reward = round(total_reward, 2)
        print('rewards: ', c_reward, nc_reward, d_reward, total_reward)

        return total_reward, done, info



    def find_collision(self, walker_list, range):
        collision = near_collision = False
        walker = None
        distance = 100.0

        ego_vehicle_trans = self.get_transform(id = self.config.ev_id, type = 'vehicle')

        walker_list = [w for w in walker_list if euclidean_distance(source_transform = (self.get_transform(id = w, type = 'walker')), destination_transform = ego_vehicle_trans) <= range]

        # iterate over the list
        for target_walker in walker_list:
            # check if walker is on driving/crossing lane
            lane_id = traci.person.getRoadID(target_walker)
            if lane_id not in str(CROSSINGS):
                continue

            w_trans = self.get_transform(id = target_walker, type = 'walker')

            # check for collision
            walker_dist = euclidean_distance(source_transform = w_trans, destination_transform = ego_vehicle_trans)
            target_walker_relative_position = compute_relative_position(source_transform = ego_vehicle_trans, destination_transform = w_trans)
            
            #if walker_dist <= 2.0:
            if (target_walker_relative_position[0] >= -4.5 and target_walker_relative_position[0] <= 1.5)\
                and (target_walker_relative_position[1] >= -2.3 and target_walker_relative_position[1] <= 2.3):
                collision = True
                walker = target_walker
                return (collision, near_collision, walker, distance)

            # check for near collision
            walker_dist -= 2.5
            
            if(target_walker_relative_position[1] >= -3.5 and target_walker_relative_position[1] <= 3.5): # same lane
                if distance > walker_dist:
                    near_collision = True
                    distance = walker_dist
                    walker = target_walker

            elif (target_walker_relative_position[1] >= -10.0 and target_walker_relative_position[1] <= 3.5) and \
                ray_intersection(p1 = w_trans, p2 = ego_vehicle_trans, n1 = self.get_forward_vector(trans = w_trans), n2 = self.get_forward_vector(trans = ego_vehicle_trans)): # next lane
                if distance > walker_dist:
                    near_collision = True
                    distance = walker_dist
                    walker = target_walker


        return (collision, near_collision, walker, distance)
             

    def get_walker_list(self):
        walker_list = []
        edges = traci.edge.getIDList()
        for edge in edges:
            walkers = traci.edge.getLastStepPersonIDs(edge)
            for w in walkers:
                walker_list.append(w)

        return walker_list


    def get_forward_vector(self, trans):
        n = ForwardVector()
        theta = trans.theta
        n.x = trans.x + 100.0 * math.cos( math.radians(trans.theta) )
        n.y = trans.y + 100.0 * math.sin( math.radians(trans.theta) )
        return n


    def generate_walker_trips(self, start_time = 0.0, end_time = 100.0, period = 0.1, pedestrians = 150):
        num_of_ped = randint(pedestrians, pedestrians + 10)
        if pedestrians == 0:
            num_of_ped = 1
        period = ((end_time - start_time) / num_of_ped)

        net = os.path.join(self.base_dir, 'sumo_configs', 'urban.net.xml')
        output_file = os.path.join(self.base_dir, 'sumo_configs', 'pedestrian.trips.xml')
        weights_file = os.path.join(self.base_dir, 'sumo_configs', 'weights_outprefix')

        randomTrips.main(randomTrips.get_options([
            '--net-file', net,
            '--output-trip-file', output_file,
            '--pedestrians',
            '--trip-attributes', 'departPos="random" arrivalPos="random" speed="2.0" ',
            '--weights-prefix', weights_file,
            '--max-distance', str(100.0),
            '-b', str(start_time),
            '-e', str(end_time),
            '-p', str(period)])
        )
