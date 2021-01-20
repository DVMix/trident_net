import pybullet as p
from utilities import _utils_v2 as utils
from utilities._utils_v2 import xyz_v2, ROBOT
from .robot_cfg_initializer import initializer

class builder:
    def __init__(self, init_cls, base_position):
        super(builder, self).__init__()
        self.cfgs          = init_cls.cfgs
        self.VARIABLE      = init_cls.VARIABLE
        self.m2m           = init_cls.m2m
        self.d2m           = init_cls.d2m
        self.m2d           = init_cls.m2d
        self.STATIC        = init_cls.STATIC
        self.structure     = init_cls.structure
        self.base_position = base_position
    # =====================================================================================================
    def BUILD_TORSO(self):
        self.M_Pelvis = ROBOT(
            Name       = 'M_Pelvis',
            CSI        = utils.geom_shape(p.GEOM_SPHERE, radius = self.VARIABLE), 
            Masses     = self.cfgs.Pelvis.mass, 
            Positions  = xyz_v2(self.d2m,self.base_position,self.cfgs.Pelvis),
            indices    = 0, 
            jntTypes   = p.JOINT_PRISMATIC, 
            axis       = self.STATIC,
            ang_speed  = None,
            angles     = None,
            torque     = None
        )

        self.J_TorsoS, self.M_TorsoS = self.structure('TorsoS',1,2)
        self.J_TorsoF, self.M_TorsoF = self.structure('TorsoF',3,4)
        self.J_TorsoR, self.M_TorsoR = self.structure('TorsoR',5,6)
    # =====================================================================================================
    def BUILD_NECK_AND_HEAD(self):
        self.J_Neck,  self.M_Neck  = self.structure('Neck',7,8)
        self.J_HeadR, self.M_HeadR = self.structure('HeadR',9,10)
        self.J_HeadF, self.M_HeadF = self.structure('HeadF',11,12)
    # =====================================================================================================
    def init_right_ARM(self): # Right Arm
        self.R_J_ShldrF, self.R_M_ShldrF = self.structure('R_ShoulderF',7,14)
        self.R_J_ShldrS, self.R_M_ShldrS = self.structure('R_ShoulderS',15,16)
        self.R_J_ElbowR, self.R_M_ElbowR = self.structure('R_ElbowR',17,18)
        self.R_J_Elbow, self.R_M_Elbow   = self.structure('R_Elbow',19,20)
        
    def init_left_ARM(self): # Left Arm
        self.L_J_ShldrF, self.L_M_ShldrF = self.structure('L_ShoulderF',7,22)
        self.L_J_ShldrS, self.L_M_ShldrS = self.structure('L_ShoulderS',23,24)
        self.L_J_ElbowR, self.L_M_ElbowR = self.structure('L_ElbowR',25,26)
        self.L_J_Elbow, self.L_M_Elbow   = self.structure('L_Elbow',27,28)
    
    def BUILD_ARMS(self):
        self.init_right_ARM()
        self.init_left_ARM()
    # =====================================================================================================
    def init_right_LEG(self): # Left Leg
        self.R_J_HipF,   self.R_M_HipF   = self.structure('R_HipF', 1,30)
        self.R_J_HipS,   self.R_M_HipS   = self.structure('R_HipS',31,32)
        self.R_J_HipR,   self.R_M_HipR   = self.structure('R_HipR',33,34)
        self.R_J_Knee,   self.R_M_Knee   = self.structure('R_Knee',35,36)
        self.R_J_AnkleF, self.R_M_AnkleF = self.structure('R_AnkleF',37,38)
        self.R_J_AnkleS, self.R_M_AnkleS = self.structure('R_AnkleS',39,40, 0.5)    
        self.R_M_Foot = ROBOT(
            Name      = 'R_M_Foot',
            CSI       = utils.geom_shape(p.GEOM_BOX, height = self.VARIABLE, coeff_P = [1,3,0.25]),  
            Masses    = self.cfgs.Foot.RIGHT.mass, 
            Positions = xyz_v2(self.m2m,self.cfgs.AnkleS.RIGHT, self.cfgs.Foot.RIGHT),
            indices   = 41, 
            jntTypes  = p.JOINT_PRISMATIC, 
            axis      = self.STATIC,
            ang_speed = None,
            angles    = None,
            torque    = None
        )
    def init_left_LEG(self): # Left Leg
        self.L_J_HipF, self.L_M_HipF = self.structure('L_HipF', 1,43)
        self.L_J_HipS, self.L_M_HipS = self.structure('L_HipS',44,45)
        self.L_J_HipR, self.L_M_HipR = self.structure('L_HipR',46,47)
        self.L_J_Knee, self.L_M_Knee = self.structure('L_Knee',48,49)
        self.L_J_AnkleF, self.L_M_AnkleF = self.structure('L_AnkleF',50,51)
        self.L_J_AnkleS, self.L_M_AnkleS = self.structure('L_AnkleS',52,53, 0.5)
        self.L_M_Foot = ROBOT(
            Name      = 'L_M_Foot',
            CSI       = utils.geom_shape(p.GEOM_BOX, height = self.VARIABLE, coeff_P = [1,3,0.25]),  
            Masses    = self.cfgs.Foot.LEFT.mass, 
            Positions = xyz_v2(self.m2m,self.cfgs.AnkleS.LEFT, self.cfgs.Foot.LEFT),
            indices   = 54, 
            jntTypes  = p.JOINT_PRISMATIC, 
            axis      = self.STATIC,
            ang_speed = None,
            angles    = None,
            torque    = None
        )
    def BUILD_LEGS(self):
        self.init_right_LEG()
        self.init_left_LEG()
    # =====================================================================================================
    def BUILD_BODY(self):
        self.BUILD_TORSO()
        self.BUILD_NECK_AND_HEAD()
        self.BUILD_ARMS()
        self.BUILD_LEGS()
        return [ 
            self.M_Pelvis, 
            self.J_TorsoS,   self.M_TorsoS,   self.J_TorsoF,   self.M_TorsoF,   self.J_TorsoR,   self.M_TorsoR,
            self.J_Neck,     self.M_Neck,     
            self.J_HeadR,    self.M_HeadR,    
            self.J_HeadF,    self.M_HeadF,
            self.R_J_ShldrF, self.R_M_ShldrF, self.R_J_ShldrS, self.R_M_ShldrS, 
            self.R_J_ElbowR, self.R_M_ElbowR, self.R_J_Elbow,  self.R_M_Elbow,
            self.L_J_ShldrF, self.L_M_ShldrF, self.L_J_ShldrS, self.L_M_ShldrS, 
            self.L_J_ElbowR, self.L_M_ElbowR, self.L_J_Elbow,  self.L_M_Elbow,
            self.R_J_HipF,   self.R_M_HipF,   self.R_J_HipS,   self.R_M_HipS,   self.R_J_HipR,   self.R_M_HipR,   
            self.R_J_Knee,   self.R_M_Knee, 
            self.R_J_AnkleF, self.R_M_AnkleF, self.R_J_AnkleS, self.R_M_AnkleS, 
            self.R_M_Foot,   
            self.L_J_HipF,   self.L_M_HipF,   self.L_J_HipS,   self.L_M_HipS,   self.L_J_HipR,   self.L_M_HipR,   
            self.L_J_Knee,   self.L_M_Knee,
            self.L_J_AnkleF, self.L_M_AnkleF, self.L_J_AnkleS, self.L_M_AnkleS, 
            self.L_M_Foot
        ]