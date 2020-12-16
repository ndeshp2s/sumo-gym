import os
import sys
import gym

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

class SumoGym(gym.Env):
    def __init__(self):
        self.sumo_running = False
        self.sumo_config = None


    def start_sumo(self):
        if not self.sumo_running:
            traci.start(self.sumo_cmd)
            self.sumo_running = True

    
    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            sys.stdout.flush()
            self.sumo_running = False


    def add_ego_vehicle(self, pose = 0.0):
        dt = traci.simulation.getDeltaT()
        vehicles = traci.vehicle.getIDList()
        for i in range(len(vehicles)):
            if vehicles[i] == self.ev.id:
                try:
                    traci.vehicle.remove(self.ev.id)
                except:
                    pass

        traci.vehicle.addFull(self.config.ev_id, 'routeEgo', depart=None, departPos=str(pose), departSpeed='0', typeID='vType0')
        traci.vehicle.setSpeedMode(self.config.ev_id, int('00000',0))
        traci.vehicle.setSpeed(self.config.ev_id, 0.0)

        traci.vehicle.subscribe(self.config.ev_id, [traci.constants.VAR_SPEED])


    def get_ev_speed(self):
        return traci.vehicle.getSubscriptionResults(self.config.ev_id)