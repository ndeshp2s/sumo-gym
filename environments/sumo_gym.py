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
from utils.misc import Transform

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

        #traci.vehicle.subscribe(self.config.ev_id, [traci.constants.VAR_SPEED])
        traci.vehicle.subscribe(self.config.ev_id, [
            traci.constants.VAR_TYPE, traci.constants.VAR_VEHICLECLASS, traci.constants.VAR_COLOR,
            traci.constants.VAR_LENGTH, traci.constants.VAR_WIDTH, traci.constants.VAR_HEIGHT,
            traci.constants.VAR_POSITION3D, traci.constants.VAR_ANGLE, traci.constants.VAR_SLOPE,
            traci.constants.VAR_SPEED, traci.constants.VAR_SPEED_LAT, traci.constants.VAR_SIGNALS
        ])


    def get_ev_speed(self):
        #return traci.vehicle.getSubscriptionResults(self.config.ev_id)
        return round(traci.vehicle.getSpeed(self.config.ev_id), 2)


    def get_transform(self, id, type):
        trans = Transform()

        if type == 'walker':
            trans.x, trans.y = traci.person.getPosition(id)
            trans.theta = 90 - traci.person.getAngle(id)
            trans.x = round(trans.x, 2)
            trans.y = round(trans.y, 2)
            trans.theta = round(trans.theta, 2)

        elif type == 'vehicle':
            trans.x, trans.y = traci.vehicle.getPosition(id)
            trans.theta = 90 - traci.vehicle.getAngle(id)
            trans.x = round(trans.x, 2)
            trans.y = round(trans.y, 2)
            trans.theta = round(trans.theta, 2)

        return trans
