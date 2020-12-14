import pybullet as p
import time
import numpy as np
import os
import sys
from collections import namedtuple

import _utils_v2 as utils
from _utils_v2 import geom_shape, xyz_v2, s, CFG_Loader
from builder.robot_cfg_initializer import initializer
from builder.robot_builder import builder
from builder.main import main

ROBOT = namedtuple('Body',['CSI', 'Masses', 'Positions', 'indices', 'jntTypes', 'axis', 'angles'])

class name(object):
    def __init__(self, 
                 path2cfgs       = './cfgs', 
                 VARIABLE        = 0.04,
                 fig_joint_coeff = 0.25,
                 centered        = False
                ):
        
        super(name, self).__init__()
        self.m2j    = ['m','j']
        self.j2m    = ['j','m']
        self.m2m    = ['m','m']
        self.d2m    = ['d','m']
        self.m2d    = ['m','d']
        self.STATIC = [0,0,0]
        
        self.centered       = centered
        self.j              = fig_joint_coeff
        self.VARIABLE       = VARIABLE
        self.body_Mass      = 20000 if centered else 1e-5
        self.base_position  = utils.coordinates(0,0,0) if centered else utils.coordinates(0,0,1)
        self.initializer    = initializer(path2cfgs, self.VARIABLE, self.j)
        self.builder        = builder(self.initializer, self.base_position)
        
    def start_enviroment(self):
        utils.init_window()
        self.main = main(self.builder, self.body_Mass, self.centered)
        self.motor_indexes, self.minmax_angles = self.main.create_body()
        
    def reset(self):
        p.resetBasePositionAndOrientation(1, list(self.base_position), [0,0,0,1])
        first_state = [0]*len(self.motor_indexes)
        
        utils.resetCoordinates(1, self.mtrIdx, first_state)
        return first_state
    
    def step(action):
        next_state = clamp(action)
        utils.resetCoordinates(1, self.mtrIdx, next_state)
        reward = 0
        game_over = True
        additional_info = ''
        return next_state, reward, game_over, additional_info
        
    @staticmethod
    def stop_enviroment():
        p.disconnect()