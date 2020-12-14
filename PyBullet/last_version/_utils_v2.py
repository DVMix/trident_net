import pybullet as p
import time
import numpy as np

from collections import namedtuple

import yaml
from easydict import EasyDict as edict
    
from yaml import load, dump
import os

coordinates = namedtuple('Coord', ['X','Y','Z'])
ROBOT = namedtuple('Body',['Name','CSI','Masses','Positions','indices','jntTypes','axis','ang_speed','angles','torque'])

#===============================================================================================================================
def load_data(path):
    files = [file.split('.')[0] for file in os.listdir(path) if '.yaml' in file]
    ndict = edict()
    for file_name in files:
        exec("%s = %d" % (file_name,0))
        with open('{}/{}.yaml'.format(path, file_name), 'r') as stream:
            globals()[file_name] = edict(yaml.full_load(stream))
        for key in globals()[file_name].keys():
            ndict[key] = globals()[file_name][key]
    return ndict

class CFG_Loader:
    def __init__(self, path2data):
        super(CFG_Loader, self).__init__()
        self.path  = path2data
        
    def get_result(self):
        return load_data(self.path)
#===============================================================================================================================
def geom_shape(shape, radius = 0.01, height = 0.01, coeff_R = 1, coeff_H = 1, coeff_P = [1,1,1]):
    if shape == p.GEOM_BOX:
        return p.createCollisionShape(shape,
                                      halfExtents = [coeff_P[0]*height, coeff_P[1]*height, coeff_P[2]*height])
    if shape == p.GEOM_SPHERE:
        return p.createCollisionShape(shape,
                                      radius = coeff_R * radius)
    if shape == p.GEOM_CYLINDER:
        return p.createCollisionShape(shape,
                                      height = coeff_H * height,
                                      radius = coeff_R * radius)
    if shape == p.GEOM_CAPSULE:
        return p.createCollisionShape(shape,
                                      height = coeff_H * height,
                                      radius = coeff_R * radius)
#=============================================================================================================================== 
def s(kind,module_name,data=None):
    if kind == 'j':
        return JOINTS[module_name].joint_coords
    if kind == 'm':
        return MASSES[module_name].center_mass_coords
    if kind == 'd':
        assert data != None, 'Data must be passed.'
        return data

def xyz(mode, module_name_M,module_name_J, data=None):
    X = s(mode[1], module_name_J, data).X - s(mode[0],module_name_M, data).X
    Y = s(mode[1], module_name_J, data).Y - s(mode[0],module_name_M, data).Y
    Z = s(mode[1], module_name_J, data).Z - s(mode[0],module_name_M, data).Z
    return [X,Y,Z]
#===============================================================================================================================
def s_v2(kind,cfg):
    if kind == 'j':
        return cfg.joint_coord 
    if kind == 'm':
        try:
            return cfg.center_of_mass
        except:
            print(cfg)
    if kind == 'd':
        return cfg

def xyz_v2(mode, BASE, STICK = None):
    if STICK is None:
        STICK = BASE
    X = s_v2(mode[1],STICK).X - s_v2(mode[0],BASE).X 
    Y = s_v2(mode[1],STICK).Y - s_v2(mode[0],BASE).Y 
    Z = s_v2(mode[1],STICK).Z - s_v2(mode[0],BASE).Z 
    return [X,Y,Z]
#===============================================================================================================================
def init_window(GUI = p.GUI):
    p.connect(GUI)
    p.createCollisionShape(p.GEOM_PLANE, planeNormal = [0,0,1])
    p.createMultiBody(0,0)
#===============================================================================================================================
def resetCamera(cameraDistance       = 5.0,
                cameraYaw            = 50, 
                cameraPitch          = -35, 
                cameraTargetPosition = [0.0, 0.0, 0.0]):
    
    p.resetDebugVisualizerCamera(
        cameraDistance       = cameraDistance, 
        cameraYaw            = cameraYaw, 
        cameraPitch          = cameraPitch, 
        cameraTargetPosition = cameraTargetPosition
    )
#===============================================================================================================================
def rad2deg(radians):
    return (radians/np.pi)* 180

def deg2rad(degrees):
    return (degrees/180) * np.pi
#===============================================================================================================================
def Mot_Pos_Ctrl(multi_body, motor_index, degrees, force = 1000, maxVelocity = 3):
    return p.setJointMotorControl2(
        multi_body,
        motor_index,
        p.POSITION_CONTROL,
        targetPosition = deg2rad(degrees),
        force          = force,
        maxVelocity    = deg2rad(maxVelocity)
    )

def resetCoordinates(multibody, motorIndexes, linkValues, force, velocity):
    assert len(motorIndexes)==len(linkValues)
    for i in range(len(motorIndexes)):
        Mot_Pos_Ctrl(multibody,motorIndexes[i],linkValues[i], force[i], velocity[i])
#===============================================================================================================================

if __name__ == '__main__':
    path = './cfgs'
    cfgs = CFG_Loader(path2data=path)