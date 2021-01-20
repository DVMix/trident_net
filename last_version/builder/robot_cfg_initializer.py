import os, sys
sys.path.append('../')

import pybullet as p
from collections import namedtuple

from utilities import _utils_v2 as utils
from utilities._utils_v2 import CFG_Loader, xyz_v2, ROBOT

class initializer:
    def __init__(self, path2cfgs, VARIABLE, j):
        super(initializer, self).__init__()
        
        self.cfgs = CFG_Loader(path2cfgs).get_result()
        self.m2j    = ['m','j']
        self.j2m    = ['j','m']
        self.m2m    = ['m','m']
        self.d2m    = ['d','m']
        self.m2d    = ['m','d']
        self.STATIC = [0,0,0]
        self.VARIABLE = VARIABLE
        self.j = j
    @staticmethod
    def dict2list(from_dict):
        return list(from_dict.values())
    
    def get_data(self, name):
        if name.startswith('R_'):
            name = name.replace('R_','')
            data = self.cfgs[name].RIGHT
            try:
                additional_name = data.motor_ops.stick_to
                if additional_name.startswith('R_'):
                    additional_name = additional_name.replace('R_','')
                    addit_data = self.cfgs[additional_name].RIGHT
                else:
                    addit_data = self.cfgs[additional_name]
            except:
                addit_data = None
        
        elif name.startswith('L_'):
            name = name.replace('L_','')
            data = self.cfgs[name].LEFT
            try:
                additional_name = data.motor_ops.stick_to
                if additional_name.startswith('L_'):
                    additional_name = additional_name.replace('L_','')
                    addit_data = self.cfgs[additional_name].LEFT
                else:
                    addit_data = self.cfgs[additional_name]
            except:
                addit_data = None
        else:
            data = self.cfgs[name]
            try:
                additional_name = data.motor_ops.stick_to
                addit_data = self.cfgs[additional_name]
            except:
                addit_data = None
        return data, addit_data

    def structure(self, name, index_J, index_M, coeff_R = 1):
        """
        CSI : CollisionShapeIndexes
        """
        basic_data, \
        addit_data = self.get_data(name)
        motor_ops  = basic_data.motor_ops
        
        J = ROBOT(
            Name      = name,
            CSI       = utils.geom_shape(p.GEOM_SPHERE, radius = self.VARIABLE, coeff_R = self.j), 
            Masses    = 1e-6,  
            Positions = xyz_v2(self.m2j,addit_data, basic_data),
            indices   = index_J,
            jntTypes  = p.JOINT_REVOLUTE, 
            axis      = self.dict2list(motor_ops.axes), 
            ang_speed = motor_ops.angular_speed,
            angles    = [motor_ops.min_angle, motor_ops.max_angle],
            torque    = motor_ops.torque
            
        )
        M = ROBOT(
            Name      = name,
            CSI       = utils.geom_shape(p.GEOM_SPHERE, radius = self.VARIABLE, coeff_R = coeff_R), 
            Masses    = basic_data.mass,  
            Positions = xyz_v2(self.j2m, basic_data),
            indices   = index_M, 
            jntTypes  = p.JOINT_PRISMATIC, 
            axis      = self.STATIC,
            ang_speed = None,
            angles    = None,
            torque    = None
        )
        return J,M
    
    def forward(self, data, target = None):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(data, target)