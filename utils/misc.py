import os
import shutil
import numpy as np
import math
import transforms3d


def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def create_directory(dir, recreate = True):
    if recreate:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    os.makedirs(dir)


def compute_relative_position(source_transform, destination_transform):
    d_xyz = np.array([destination_transform.x, destination_transform.y, 0])
    s_xyz = np.array([source_transform.x, source_transform.y, 0])
    pos = d_xyz - s_xyz

    pitch = math.radians(0.0)
    roll = math.radians(0.0)
    yaw = math.radians(source_transform.theta)

    R = transforms3d.euler.euler2mat(roll, pitch, yaw).T
    pos_rel = np.dot(R, pos)

    return (pos_rel[0], -pos_rel[1])


def compute_relative_heading(source_transform, destination_transform):
    d_heading = destination_transform.theta
    s_heading = source_transform.theta
    head_rel = d_heading - s_heading
    head_rel = (head_rel + 360) % 360
    head_rel = round(head_rel, 2)

    return head_rel


def get_index(val, start, stop, num):

    grids = np.linspace(start, stop, num)
    features = np.zeros(num)

    #Check extremes
    if val <= grids[0] or val > grids[-1]:
        return features, False

    for i in range(len(grids) - 1):
        if val >= grids[i] and val < grids[i + 1]:
            features[i] = 1

    return features, True


def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def euclidean_distance(source_transform, destination_transform):
    dx = source_transform.x - destination_transform.x
    dy = source_transform.y - destination_transform.y

    return math.sqrt(dx * dx + dy * dy)

def ray_intersection(p1, p2, n1, n2):
    if (n1.y * n2.x - n1.x * n2.y) == 0.0 or n2.x == 0.0:
        return False

    u = (p1.x * n2.y + p2.y * n2.x - p2.x * n2.y - p1.y * n2.x) / (n1.y * n2.x - n1.x * n2.y)
    v = (p1.x + n1.x * u - p2.x) / n2.x

    if u > 0 and v > 0:
        return True

    return False


class Transform():
    def __init__(self, x = 0, y = 0, theta = 0):
        self.x = x
        self.y = y
        self.theta = theta

class ForwardVector():
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y


def getCos(val):
    return round( math.cos(math.radians((val))), 4)

def getSin(val):
    return round( math.sin(math.radians((val))), 4)

# def compute_relative_position(ego_vehicle, pedestrian):

#     p = np.array([pedestrian.x, pedestrian.y, 1])

#     R = np.array([[getCos(ego_vehicle.theta), -getSin(ego_vehicle.theta)],
#                 [getSin(ego_vehicle.theta), getCos(ego_vehicle.theta)]])
#     R = np.transpose(R)

#     d = np.array([ego_vehicle.x, ego_vehicle.y])
#     R_d =  np.matmul(-R, d)

#     T = np.array([ [R[0][0], R[0][1], R_d[0]],
#                    [R[1][0], R[1][1], R_d[1]],
#                    [0, 0, 1]])


#     return np.matmul(T, p)