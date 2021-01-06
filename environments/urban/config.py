# Ego Vehicle related
ev_id = 'ego_vehicle'
ev_maximum_speed = 15.0
ev_target_speed = 10.0
ev_start_position = 20.0
ev_goal_position = [78, 140]

max_speed = 12 # in meter/second
target_speed = 8

# sumo related
use_gui = False

# observation space related
grid_height = 40*2
grid_width = 30*2
features = 4

grid_height_min = -4*2
grid_height_max = 35*2

grid_width_min = -15*2
grid_width_max = 14*2

# action space related
N_DISCRETE_ACTIONS = 4

# pedestrian related
safe_distance = 1.0