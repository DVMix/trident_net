import pybullet as p
import time
import numpy as np

from collections import namedtuple
#===============================================================================================================================
coordinates = namedtuple('Coord', ['X','Z','Y'])
detail_conf = namedtuple('detail_conf', ['module_name', 'mass', 'center_mass_coords'])
MASSES = dict(
    HeadR       = detail_conf('HeadR',      1.7000,coordinates(0.00, 1.61, 0.03)),
    HeadF       = detail_conf('HeadF',      4.2980,coordinates(0.00, 1.64, 0.13)),
    Neck        = detail_conf('Neck',       1.6340,coordinates(0.00, 1.51, 0.03)),
    #----------------------------------------------------------------------------
    TorsoS      = detail_conf('TorsoS',     4.1050,coordinates(0.00, 1.03,-0.01)),
    TorsoF      = detail_conf('TorsoF',     2.5000,coordinates(0.01, 1.08, 0.02)),
    TorsoR      = detail_conf('TorsoR',     20.590,coordinates(0.00, 1.29,-0.01)),
    Pelvis      = detail_conf('Pelvis',     5.5420,coordinates(0.00, 0.91,-0.01)),
    #----------------------------------------------------------------------------
    L_ShoulderF = detail_conf('L_ShoulderF',3.2390,coordinates(-0.18,1.35, 0.03)),
    L_ShoulderS = detail_conf('L_ShoulderS',3.4870,coordinates(-0.23,1.28, 0.01)),
    L_ElbowR    = detail_conf('L_ElbowR',   3.0770,coordinates(-0.27,1.14,-0.03)),
    L_Elbow     = detail_conf('L_Elbow',    3.2637,coordinates(-0.31,0.90,-0.02)),
    #----------------------------------------------------------------------------
    L_HipF      = detail_conf('L_HipF',     3.1890,coordinates(-0.10,0.89,-0.05)),
    L_HipS      = detail_conf('L_HipS',     2.5570,coordinates(-0.10,0.81,-0.03)),
    L_HipR      = detail_conf('L_HipR',     4.3837,coordinates(-0.11,0.66, 0.00)),
    L_Knee      = detail_conf('L_Knee',     6.1700,coordinates(-0.11,0.37,-0.02)),
    L_AnkleF    = detail_conf('L_AnkleF',   0.7030,coordinates(-0.12,0.10,-0.01)),
    L_AnkleS    = detail_conf('L_AnkleS',   1.5110,coordinates(-0.12,0.04, 0.02)),
    L_Foot      = detail_conf('L_Foot',     1.0000,coordinates(-0.12,0.02, 0.03)),
    #----------------------------------------------------------------------------
    R_ShoulderF = detail_conf('R_ShoulderS',3.2390,coordinates(0.18, 1.35, 0.03)),
    R_ShoulderS = detail_conf('R_ShoulderF',3.4870,coordinates(0.23, 1.28, 0.01)),
    R_ElbowR    = detail_conf('R_ElbowR',   3.0770,coordinates(0.28, 1.14,-0.03)),
    R_Elbow     = detail_conf('R_Elbow',    3.2637,coordinates(0.32, 0.90,-0.02)),
    #----------------------------------------------------------------------------
    R_HipF      = detail_conf('R_HipF',     3.1890,coordinates(0.11, 0.89,-0.05)),
    R_HipS      = detail_conf('R_HipS',     2.5570,coordinates(0.11, 0.81,-0.03)),
    R_HipR      = detail_conf('R_HipR',     4.3837,coordinates(0.11, 0.66, 0.00)),
    R_Knee      = detail_conf('R_Knee',     6.1700,coordinates(0.11, 0.37,-0.02)),
    R_AnkleF    = detail_conf('R_AnkleF',   0.7030,coordinates(0.11, 0.10,-0.01)),
    R_AnkleS    = detail_conf('R_AnkleS',   1.5110,coordinates(0.11, 0.04, 0.02)),
    R_Foot      = detail_conf('R_Foot',     1.0000,coordinates(0.11, 0.02, 0.03)),
)
#===============================================================================================================================
join_conf = namedtuple('join_conf', ['module_name', 'joint_coords'])
JOINTS = dict(
    HeadR       = join_conf('HeadR',       coordinates(0.00,  1.54,  0.02)),
    HeadF       = join_conf('HeadF',       coordinates(0.00,  1.64,  0.04)),
    Neck        = join_conf('Neck',        coordinates(0.01,  1.48,  0.04)),
    #----------------------------------------------------------------------
    TorsoF      = join_conf('TorsoF',      coordinates(0.00,  1.03,  0.01)),
    TorsoS      = join_conf('TorsoS',      coordinates(0.00,  1.03,  0.01)),
    TorsoR      = join_conf('TorsoR',      coordinates(0.00,  1.23,  0.02)),
    #----------------------------------------------------------------------
    L_ShoulderF = join_conf('L_ShoulderF', coordinates(-0.09, 1.35,  0.02)),
    L_ShoulderS = join_conf('L_ShoulderS', coordinates(-0.21, 1.35,  0.03)),
    L_ElbowR    = join_conf('L_ElbowR',    coordinates(-0.25, 1.21, -0.01)),
    L_Elbow     = join_conf('L_Elbow',     coordinates(-0.28, 1.08, -0.03)),
    #----------------------------------------------------------------------
    L_HipF      = join_conf('L_HipF',      coordinates(-0.10, 0.89,  0.00)),
    L_HipS      = join_conf('L_HipS',      coordinates(-0.10, 0.89,  0.00)),
    L_HipR      = join_conf('L_HipR',      coordinates(-0.10, 0.73,  0.00)),
    L_Knee      = join_conf('L_Knee',      coordinates(-0.11, 0.49,  0.00)),
    L_AnkleF    = join_conf('L_AnkleF',    coordinates(-0.11, 0.10,  0.00)),
    L_AnkleS    = join_conf('L_AnkleS',    coordinates(-0.11, 0.10,  0.00)),
    #----------------------------------------------------------------------
    R_ShoulderF = join_conf('R_ShoulderF', coordinates(0.10,  1.35,  0.03)),
    R_ShoulderS = join_conf('R_ShoulderS', coordinates(0.21,  1.35,  0.03)),
    R_ElbowR    = join_conf('R_ElbowR',    coordinates(0.26,  1.21, -0.01)),
    R_Elbow     = join_conf('R_Elbow',     coordinates(0.29,  1.08, -0.03)),
    #----------------------------------------------------------------------
    R_HipF      = join_conf('R_HipF',      coordinates(0.11,  0.89,  0.00)),
    R_HipS      = join_conf('R_HipS',      coordinates(0.11,  0.89,  0.00)),
    R_HipR      = join_conf('R_HipR',      coordinates(0.11,  0.73,  0.00)),
    R_Knee      = join_conf('R_Knee',      coordinates(0.11,  0.49,  0.00)),
    R_AnkleF    = join_conf('R_AnkleF',    coordinates(0.11,  0.10,  0.00)),
    R_AnkleS    = join_conf('R_AnkleS',    coordinates(0.11,  0.10,  0.00)),
)
#===============================================================================================================================
motor_optns = namedtuple('motor_optns',['motor_name', 'sticks_to', 'axes', 
                                          'min_angle','max_angle', 'angular_speed', 'torque'])
MOTORS = dict(
    HeadF       = motor_optns('HeadF',      'HeadR',       coordinates(0.0,-1.0, 0.0), -60, 20,130,450.0),
    HeadR       = motor_optns('HeadR',      'Neck',        coordinates(0.0, 0.0,-1.0), -80, 80,420,172.0),
    Neck        = motor_optns('Neck',       'TorsoR',      coordinates(0.0,-1.0, 0.0), -10, 40,110,360.0),
    #----------------------------------------------------------------------------------------------------
    TorsoF      = motor_optns('TorsoF',     'TorsoS',      coordinates(0.0,-1.0, 0.0), -12, 40, 75,525.0),
    TorsoS      = motor_optns('TorsoS',     'Pelvis',      coordinates(1.0, 0.0, 0.0), -15, 15,753, 75.0),
    TorsoR      = motor_optns('TorsoR',     'TorsoF',      coordinates(0.0, 0.0,-1.0), -95, 95,230,525.0),
    #----------------------------------------------------------------------------------------------------
    R_HipF      = motor_optns('R_HipF',     'Pelvis',      coordinates(0.0,-1.0, 0.0), -90, 35,350,450.0),
    R_HipS      = motor_optns('R_HipS',     'R_HipF',      coordinates(1.0, 0.0, 0.0), -90, 12, 70,345.0),
    R_HipR      = motor_optns('R_HipR',     'R_HipS',      coordinates(0.0, 0.0,-1.0), -90, 90,113,240.0),
    R_Knee      = motor_optns('R_Knee',     'R_HipR',      coordinates(0.0,-1.0, 0.0),   0,110,455,375.0),
    R_AnkleF    = motor_optns('R_AnkleF',   'R_Knee',      coordinates(0.0,-1.0, 0.0), -40, 25,180,240.0),
    R_AnkleS    = motor_optns('R_AnkleS',   'R_AnkleF',    coordinates(1.0, 0.0, 0.0), -30, 30,210,150.0),
    #----------------------------------------------------------------------------------------------------
    R_ShoulderF = motor_optns('R_ShoulderF','TorsoR',      coordinates(0.0,-1.0, 0.0),-150, 90,360,322.5),
    R_ShoulderS = motor_optns('R_ShoulderS','R_SHoulderF', coordinates(1.0, 0.0, 0.0),-150,  0,180,495.0),
    R_ElbowR    = motor_optns('R_ElbowR',   'R_ShoulderS', coordinates(0.0, 0.0,-1.0),-110,110,360,225.0),
    R_Elbow     = motor_optns('R_Elbow',    'R_ElbowR',    coordinates(0.0,-1.0, 0.0),-120,  5,230,255.0),
    R_WristF    = motor_optns('R_WristF',   'R_WristS',    coordinates(0.0,-1.0, 0.0), -18, 33,100, 37.5),
    R_WristS    = motor_optns('R_WristS',   'R_WristR',    coordinates(1.0, 0.0, 0.0), -25, 15,100, 37.5),
    R_WristR    = motor_optns('R_WristR',   'R_Elbow',     coordinates(0.0, 0.0,-1.0),-110,110,280, 75.0),
    #----------------------------------------------------------------------------------------------------
    L_HipF      = motor_optns('L_HipF',     'Pelvis',      coordinates(0.0,-1.0, 0.0), -90, 35,350,450.0),
    L_HipS      = motor_optns('L_HipS',     'L_HipF',      coordinates(1.0, 0.0, 0.0), -12, 90, 70,345.0),
    L_HipR      = motor_optns('L_HipR',     'L_HipS',      coordinates(0.0, 0.0,-1.0), -90, 90,113,240.0),
    L_Knee      = motor_optns('L_Knee',     'L_HipR',      coordinates(0.0,-1.0, 0.0),   0,110,455,375.0),
    L_AnkleF    = motor_optns('L_AnkleF',   'L_Knee',      coordinates(0.0,-1.0, 0.0), -40, 25,180,240.0),
    L_AnkleS    = motor_optns('L_AnkleS',   'L_AnkleF',    coordinates(1.0, 0.0, 0.0), -30, 30,210,150.0),
    #----------------------------------------------------------------------------------------------------
    L_ShoulderF = motor_optns('L_ShoulderF','TorsoR',      coordinates(0.0,-1.0, 0.0),-150, 90,360,322.5),
    L_ShoulderS = motor_optns('L_ShoulderS','L_ShoulderF', coordinates(1.0, 0.0, 0.0),   0,150,180,495.0),
    L_ElbowR    = motor_optns('L_ElbowR',   'L_ShoulderS', coordinates(0.0, 0.0,-1.0),-110,110,360,225.0),
    L_Elbow     = motor_optns('L_Elbow',    'L_ElbowR',    coordinates(0.0,-1.0, 0.0),-120,  5,230,255.0),
    L_WristF    = motor_optns('L_WristF',   'L_WristS',    coordinates(0.0,-1.0, 0.0), -18, 33,100, 37.5),
    L_WristS    = motor_optns('L_WristS',   'L_WristR',    coordinates(1.0, 0.0, 0.0), -15, 25,100, 37.5),
    L_WristR    = motor_optns('L_WristR',   'L_Elbow',     coordinates(0.0, 0.0,-1.0),-110,110,280, 75.0),
)