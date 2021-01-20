import torch

import pybullet as p
import time
import numpy as np
import os
import sys
from collections import namedtuple

from utilities import _utils_v2 as utils
from utilities._utils_v2 import geom_shape, xyz_v2, s, CFG_Loader, rad2deg, deg2rad

from builder import initializer, builder, main

from .fns import terminal_fn, reward_fn

#     state_info = p.getJointState(body_index, joint_index)

#     posOrient = p.getLinkState(body_index, joint_index)[:2]
#     print('jointPosition    = ', state_info[0])
#     print('jointVelocity    = ', state_info[1])
#     print('jointReactForces = ', state_info[2])
#     print('appliedTorque    = ', state_info[3])

#     print('linkWorldPosition    = ', posOrient[0])
#     print('linkWorldOrientation = ', posOrient[1])  

#     body_index = 1
#     motor = 1
#     # jointPosition
#     # jointVelocity
#     # jointReactionForces
#     # appliedJointMotorTorque
#     JS = p.getJointState(1, motor)
#     # linkWorldPosition
#     # linkWorldOrientation
#     # localInertialFramePosition
#     # localInertialFrameOrientation
#     # worldLinkFramePosition
#     # worldLinkFrameOrientation
#     # worldLinkLinearVelocity
#     # worldLinkAngularVelocity
#     LS = p.getLinkState(1, motor)
# def det_single_motor_state(body_index, motor):
#     array = []
#     for source in [p.getJointState(body_index, motor), 
#                    p.getLinkState(body_index, motor)[:2]]:
#         for i in source:
#             if not isinstance(i, (list, tuple)):
#                 array.append(i)
#             else:
#                 array.extend(i)
#     return array

# def get_state(body_index, motor_list):
#     array = []
#     for motor in motor_list:
#         state = det_single_motor_state(body_index, motor)
#         array.extend(state)

#     return torch.tensor(data = array, dtype = torch.float32)

def det_single_motor_state(body_index, motor):
    array = []
    jointState = p.getJointState(body_index, motor)
    jointState = jointState[0],jointState[1],jointState[3]
    linkState  = p.getLinkState(body_index, motor)[:2]
    for source in [jointState, linkState]:
        for i in source:
            if not isinstance(i, (list, tuple)):
                array.append(i)
            else:
                array.extend(i)
    return array

def get_state(body_index, motor_list):
    array = []
    for motor in motor_list:
        state = det_single_motor_state(body_index, motor)
        array.extend(state)

    return torch.tensor(data = array, dtype = torch.float32).reshape(1,len(motor_list),-1)

class daughter:
    def __init__(self, 
                 path2cfgs, 
                 angle_step      = None,
                 RTS             = 1,
                 VARIABLE        = 0.04,
                 fig_joint_coeff = 0.25,
                 centered        = False):
        # super(daughter, self).__init__()
        
        self.path2cfgs = path2cfgs
        self.VARIABLE  = VARIABLE
        self.joint_cf  = fig_joint_coeff
        self.centered  = centered 
        self.body_Mass = 200 if centered else 1e-5
        self.position  = utils.coordinates(0,0,0) if centered else utils.coordinates(0,0,1)
        self.init_cfgs = initializer(self.path2cfgs, self.VARIABLE, self.joint_cf)
        self.angle_step= angle_step if angle_step is not None else 1
        self.RTS       = RTS   
        
    def init_enviroment(self, mode = p.GUI):
        p.connect(mode)# Avalible p.GUI p.DIRECT
        p.createCollisionShape(p.GEOM_PLANE, planeNormal = [0,0,1])
        p.createMultiBody(0,0)
        
        self.blocks = builder(self.init_cfgs, self.position)
        self.body   = main(self.blocks, self.body_Mass, self.centered)
        
        self.minmax   = [data.angles for data in self.body.cfgs if data.jntTypes==0]
        self.mtrIdx   = [i for i, data in enumerate(self.body.cfgs) if data.jntTypes==0]
        self.force    = [data.torque for data in self.body.cfgs if data.jntTypes==0]
        self.velocity = [data.ang_speed for data in self.body.cfgs if data.jntTypes==0]
        
        self.joint_names = [(i, data.Name) for i, data in enumerate(self.body.cfgs) if data.jntTypes==0]
        self.link_names  = [(i, data.Name) for i, data in enumerate(self.body.cfgs) if data.jntTypes==1]
        
        self.body_index = 1
        self.body.create_body()
        p.setRealTimeSimulation(self.RTS)
        #Add extra lateral friction to the feets
        p.changeDynamics(1,41,lateralFriction=2)
        p.changeDynamics(1,54,lateralFriction=2)
    
    def get_names(self):
        return self.names
    
    def set_avalible_motors(self,avalible_motors):
        self.motors = avalible_motors
        self.current_state_next = get_state(self.body_index, self.motors)
        
    def close(self):
        p.disconnect()   

    def reset(self):
        if self.RTS==0:
            p.setRealTimeSimulation(1)
        p.setGravity(0,0,0)
        
        self.angles = [0]*len(self.mtrIdx)
        utils.resetCoordinates(1, self.mtrIdx, self.angles, 
                               [i * 100 for i in self.force], 
                               [i * 100 for i in self.velocity]) 
        
        time.sleep(1)
        p.resetBasePositionAndOrientation(1, [0, 0, 1.1], [0, 0, 0, 1])
        p.setGravity(0,0,-9.81)
        
        time.sleep(0.5)
        if self.RTS==0:
            p.setRealTimeSimulation(0)
    
    def get_real_angle_state(self, body_index):
        pass
        
    def step(self, action):
        motor_idx = int(action/2)
        sign = 1 if action % 2 == 0 else -1
        global_mtr_index = self.mtrIdx.index(self.motors[motor_idx])
        
        pred_angle = self.angles[global_mtr_index]+sign*self.angle_step
        if (pred_angle >= self.minmax[global_mtr_index][0]) and (pred_angle <= self.minmax[global_mtr_index][1]):
            self.angles[global_mtr_index] = pred_angle
        # self.angles = get_real_angle_state(self.body_index)
        previous_position, _ = p.getBasePositionAndOrientation(self.body_index)
        
        p.setJointMotorControl2(
            self.body_index, 
            self.motors[motor_idx],
            p.POSITION_CONTROL,
            targetPosition = deg2rad(self.angles[global_mtr_index]),
            maxVelocity    = deg2rad(self.velocity[global_mtr_index]), 
            force          = self.force[global_mtr_index]
        )
        if self.RTS==0:
            p.stepSimulation()
        current_position, _ = p.getBasePositionAndOrientation(self.body_index)
        state_next = get_state(self.body_index, self.motors)
        
        # terminal = True if b_position[2] < 0.6 else False
        terminal = terminal_fn(current_position, 
                               self.angles[global_mtr_index], 
                               self.minmax[global_mtr_index])
        # reward = 1
        reward = reward_fn(previous_position, current_position, 
                           self.angles[global_mtr_index], 
                           self.minmax[global_mtr_index])
        
        info = current_position
        return state_next, reward, terminal, info