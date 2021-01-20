import math 
import numpy as np

def terminal_fn(current_position, angle, minmax):
    terminal = False if current_position[2] > 0.75 else True 
    ang_terminal = False if (angle>=minmax[0])&(angle<=minmax[1]) else True
    terminal = terminal or ang_terminal
    return terminal

def reward_fn(previous_position, current_position, angle, minmax):
    zero__position =  np.power(100*(current_position[0]-previous_position[0]),2)
    first_position =  np.power(100*(current_position[1]-previous_position[1]),2)
    
#     print(dict(zero_position=zero__position, first_position=first_position))
#     angular_reward = 360
#     if angle<minmax[0]:
#         angular_reward += (angle - minmax[0])
#     if angle>minmax[1]:
#         angular_reward += (minmax[1] - angle)
#     angular_reward/=360
    if (angle>=minmax[0])&(angle<=minmax[1]):
        angular_reward = 1
    else:
        angular_reward = -1
        
    dekart_distance = np.power(zero__position+first_position,0.5)
    reward = dekart_distance + angular_reward
    #if (reward != 1) and (reward != -1):
    #    print(dict(reward=reward, dekart_distance=dekart_distance))
    #return reward
    return reward, dekart_distance

def rad2deg(radians: float) -> float:
    degree = radians*180/math.pi
    return degree

def deg2rad(degree: float) -> float:
    radians = degree*math.pi/180
    return radians

def midle_foot_coord(position):
    avg_pos = (position[0] + position[1])/2
    return avg_pos

def landed_foot(foot_idx_list = [41,54], threshold = 0.1):
    link_states = p.getLinkStates(1, foot_idx_list,1,1)
    position = np.array([np.array(tuple_[0]) for tuple_ in link_states])
    mask = [l[2]<threshold for l in position]
    position = position[mask]
    
    if position.shape[0]==2:
        position = midle_foot_coord(position)
    else:
        position = position[0]
    return position

def coord2angle_N(point1: tuple, point2: tuple) -> tuple:
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]
    z = point2[2] - point1[2]
    
    xd = math.sqrt(math.pow(x,2)+math.pow(z,2))
    x_cos = z/xd
    x_arccos= math.acos(x_cos)
    x_angle = rad2deg(x_arccos)
    
    yd = math.sqrt(math.pow(y,2)+math.pow(z,2))
    y_cos = z/yd
    y_arccos= math.acos(y_cos)
    y_angle = rad2deg(y_arccos)
    
    xyd  = math.sqrt(math.pow(x,2)+math.pow(y,2))
    xyzd = math.sqrt(math.pow(z,2)+math.pow(xyd,2))
    xyz_cos   = z/xyzd
    xyz_arccos= math.acos(xyz_cos)
    xyz_angle = rad2deg(xyz_arccos)
    
    return round(x_angle,1),round(y_angle,1),round(xyz_angle,1)