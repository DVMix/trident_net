import pybullet as p
import time
import numpy as np
import os
import _utils as u
from _utils import geom_shape, xyz, s
import sys
from collections import namedtuple

class ROBOT(object):
    def __init__(self):
        super(ROBOT, self).__init__()
        self.hello_world = 'Hello, World!'
        self.init_body()
        
    def init_body(self):
        self.load_cfgs()
        pass
    
    def load_cfgs(self, cfgs):
        pass
    
    def move_body(self, command):
        pass