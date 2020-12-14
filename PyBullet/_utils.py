import pybullet as p
import time
import numpy as np

from collections import namedtuple
print("""#================================================================================================
#Fedor CFG file:                                                                                #
#    Using namedtuples for humanreadable view:                                                  #
#                                                                                               #
#    coordinates = namedtuple('Coord', ['X','Y','Z'])                                           #
#    detail_conf = namedtuple('detail_conf', ['module_name',  'mass', 'center_mass_coords'])    #
#    join_conf   = namedtuple('join_conf',   ['module_name',  'joint_coords'])                  #
#    motor_optns = namedtuple('motor_optns', ['motor_name',   'sticks_to', 'axes', 'min_angel', #
#                                             'max_angel',    'angular_speed','torque'])        #                              
#================================================================================================

Base: Block = namedtuple('Building_Block', ['DETAIL_CFG'       (MASS) , 
                                            'JOINT_COORDINATE' (JOINT), 
                                            'MOTOR_CFG'        (MOTOR)])
""")

def print_info():
    print('''
    MASS = [MASS_HeadR,MASS_HeadF,MASS_Neck,MASS_TorsoS,MASS_TorsoF,MASS_TorsoR,MASS_Pelvis,MASS_L_ShoulderF,
MASS_L_ShoulderS,MASS_L_ElbowR,MASS_L_Elbow,MASS_L_HipF,MASS_L_HipS,MASS_L_HipR,MASS_L_Knee,MASS_L_AnkleF,
MASS_L_AnkleS,MASS_L_Foot,MASS_L_Foot_FL,MASS_L_Foot_FB,MASS_R_ShoulderF,MASS_R_ShoulderS,MASS_R_ElbowR,
MASS_R_Elbow,MASS_R_HipF,MASS_R_HipS,MASS_R_HipR,MASS_R_Knee,MASS_R_AnkleF,MASS_R_AnkleS,MASS_R_Foot,
MASS_R_Foot_FL,MASS_R_Foot_FB]

JOINT = [JOINT_HeadR, JOINT_HeadF, JOINT_Neck, JOINT_TorsoS, JOINT_TorsoF, JOINT_TorsoR, JOINT_L_ShoulderF,
JOINT_L_ShoulderS,JOINT_L_ElbowR,JOINT_L_Elbow,JOINT_L_HipF,JOINT_L_HipS,JOINT_L_HipR,JOINT_L_Knee, 
JOINT_L_AnkleF, JOINT_L_AnkleS, JOINT_R_ShoulderF, JOINT_R_ShoulderS, JOINT_R_ElbowR, JOINT_R_Elbow, JOINT_R_HipF,
JOINT_R_HipS, JOINT_R_HipR, JOINT_R_Knee, JOINT_R_AnkleF, JOINT_R_AnkleS]

MOTOR = [MOTOR_HeadF, MOTOR_HeadR, MOTOR_Neck, MOTOR_TorsoR, MOTOR_TorsoF, MOTOR_TorsoS, MOTOR_R_AnkleF,
MOTOR_R_AnkleS, MOTOR_R_Knee, MOTOR_R_ElbowR, MOTOR_R_ShoulderF, MOTOR_R_Finger_Index, MOTOR_R_ShoulderS, 
MOTOR_R_WristS, MOTOR_R_WristF, MOTOR_R_Finger_Middle, MOTOR_R_HipS, MOTOR_R_Finger_ThumbS, MOTOR_R_Elbow,
MOTOR_R_HipR, MOTOR_R_HipF, MOTOR_R_Finger_Ring, MOTOR_R_Finger_Little, MOTOR_R_Finger_Thumb, MOTOR_R_WristR,
MOTOR_L_AnkleS, MOTOR_L_AnkleF, MOTOR_L_Knee, MOTOR_L_ElbowR, MOTOR_L_ShoulderF, MOTOR_L_Finger_Index, MOTOR_L_ShoulderS,
MOTOR_L_WristS, MOTOR_L_WristF, MOTOR_L_HipS, MOTOR_L_Finger_Middle, MOTOR_L_Finger_ThumbS, MOTOR_L_HipR, MOTOR_L_Elbow,
MOTOR_L_HipF, MOTOR_L_Finger_Ring, MOTOR_L_Finger_Little, MOTOR_L_Finger_Thumb, MOTOR_L_WristR]
    ''')
coordinates = namedtuple('Coord', ['X','Y','Z'])
detail_conf = namedtuple('detail_conf', ['module_name', 'mass', 'center_mass_coords'])

MASS_HeadR       = detail_conf('HeadR',      1.7000,coordinates(0.00, 1.61, 0.03))
MASS_HeadF       = detail_conf('HeadF',      4.2980,coordinates(0.00, 1.64, 0.13))
MASS_Neck        = detail_conf('Neck',       1.6340,coordinates(0.00, 1.51, 0.03))

MASS_TorsoS      = detail_conf('TorsoS',     4.1050,coordinates(0.00, 1.03,-0.01))
MASS_TorsoF      = detail_conf('TorsoF',     2.5000,coordinates(0.01, 1.08, 0.02))
MASS_TorsoR      = detail_conf('TorsoR',     20.590,coordinates(0.00, 1.29,-0.01))
MASS_Pelvis      = detail_conf('Pelvis',     5.5420,coordinates(0.00, 0.91,-0.01))

MASS_L_ShoulderF = detail_conf('L_ShoulderF',3.2390,coordinates(-0.18,1.35, 0.03))
MASS_L_ShoulderS = detail_conf('L_ShoulderS',3.4870,coordinates(-0.23,1.28, 0.01))
MASS_L_ElbowR    = detail_conf('L_ElbowR',   3.0770,coordinates(-0.27,1.14,-0.03))
MASS_L_Elbow     = detail_conf('L_Elbow',    3.2637,coordinates(-0.31,0.90,-0.02))
MASS_L_HipF      = detail_conf('L_HipF',     3.1890,coordinates(-0.10,0.89,-0.05))
MASS_L_HipS      = detail_conf('L_HipS',     2.5570,coordinates(-0.10,0.81,-0.03))
MASS_L_HipR      = detail_conf('L_HipR',     4.3837,coordinates(-0.11,0.66, 0.00))
MASS_L_Knee      = detail_conf('L_Knee',     6.1700,coordinates(-0.11,0.37,-0.02))
MASS_L_AnkleF    = detail_conf('L_AnkleF',   0.7030,coordinates(-0.12,0.10,-0.01))
MASS_L_AnkleS    = detail_conf('L_AnkleS',   1.5110,coordinates(-0.12,0.04, 0.02))
MASS_L_Foot      = detail_conf('L_Foot',     1.0000,coordinates(-0.12,0.02, 0.03))
MASS_L_Foot_FL   = detail_conf('L_Foot_FL',  0.1000,coordinates(-0.14,0.02, 0.16)) # L_Foot_Finger_Little
MASS_L_Foot_FB   = detail_conf('L_Foot_FB',  0.1000,coordinates(-0.09,0.02, 0.16)) # L_Foot_Finger_Big

MASS_R_ShoulderF = detail_conf('R_ShoulderS',3.2390,coordinates(0.18, 1.35, 0.03))
MASS_R_ShoulderS = detail_conf('R_ShoulderF',3.4870,coordinates(0.23, 1.28, 0.01))
MASS_R_ElbowR    = detail_conf('R_ElbowR',   3.0770,coordinates(0.28, 1.14,-0.03))
MASS_R_Elbow     = detail_conf('R_Elbow',    3.2637,coordinates(0.32, 0.90,-0.02))
MASS_R_HipF      = detail_conf('R_HipF',     3.1890,coordinates(0.11, 0.89,-0.05))
MASS_R_HipS      = detail_conf('R_HipS',     2.5570,coordinates(0.11, 0.81,-0.03))
MASS_R_HipR      = detail_conf('R_HipR',     4.3837,coordinates(0.11, 0.66, 0.00))
MASS_R_Knee      = detail_conf('R_Knee',     6.1700,coordinates(0.11, 0.37,-0.02))
MASS_R_AnkleF    = detail_conf('R_AnkleF',   0.7030,coordinates(0.11, 0.10,-0.01))
MASS_R_AnkleS    = detail_conf('R_AnkleS',   1.5110,coordinates(0.11, 0.04, 0.02))
MASS_R_Foot      = detail_conf('R_Foot',     1.0000,coordinates(0.11, 0.02, 0.03))
MASS_R_Foot_FL   = detail_conf('R_Foot_FL',  0.1000,coordinates(13.00,0.02, 0.16)) # R_Foot_Finger_Little
MASS_R_Foot_FB   = detail_conf('R_Foot_FB',  0.1000,coordinates(0.08, 0.02, 0.16)) # R_Foot_Finger_Big


# ==========================================================================
join_conf         = namedtuple('join_conf', ['module_name', 'joint_coords'])

JOINT_HeadR       = join_conf('HeadR',       coordinates(0.00,  1.54,  0.02))
JOINT_HeadF       = join_conf('HeadF',       coordinates(0.00,  1.64,  0.04))
JOINT_Neck        = join_conf('Neck',        coordinates(0.01,  1.48,  0.04))
#--------------------------------------------------------------------------
JOINT_TorsoS      = join_conf('TorsoS',      coordinates(0.00,  1.03,  0.01))
JOINT_TorsoF      = join_conf('TorsoF',      coordinates(0.00,  1.03,  0.01))
JOINT_TorsoR      = join_conf('TorsoR',      coordinates(0.00,  1.23,  0.02))
#--------------------------------------------------------------------------
JOINT_L_ShoulderF = join_conf('L_ShoulderF', coordinates(-0.09, 1.35,  0.02))
JOINT_L_ShoulderS = join_conf('L_ShoulderS', coordinates(-0.21, 1.35,  0.03))
JOINT_L_ElbowR    = join_conf('L_ElbowR',    coordinates(-0.25, 1.21, -0.01))
JOINT_L_Elbow     = join_conf('L_Elbow',     coordinates(-0.28, 1.08, -0.03))
JOINT_L_HipF      = join_conf('L_HipF',      coordinates(-0.10, 0.89,  0.00))
JOINT_L_HipS      = join_conf('L_HipS',      coordinates(-0.10, 0.89,  0.00))
JOINT_L_HipR      = join_conf('L_HipR',      coordinates(-0.10, 0.73,  0.00))
JOINT_L_Knee      = join_conf('L_Knee',      coordinates(-0.11, 0.49,  0.00))
JOINT_L_AnkleF    = join_conf('L_AnkleF',    coordinates(-0.11, 0.10,  0.00))
JOINT_L_AnkleS    = join_conf('L_AnkleS',    coordinates(-0.11, 0.10,  0.00))
#--------------------------------------------------------------------------
JOINT_R_ShoulderF = join_conf('R_ShoulderF', coordinates(0.10,  1.35,  0.03))
JOINT_R_ShoulderS = join_conf('R_ShoulderS', coordinates(0.21,  1.35,  0.03))
JOINT_R_ElbowR    = join_conf('R_ElbowR',    coordinates(0.26,  1.21, -0.01))
JOINT_R_Elbow     = join_conf('R_Elbow',     coordinates(0.29,  1.08, -0.03))
JOINT_R_HipF      = join_conf('R_HipF',      coordinates(0.11,  0.89,  0.00))
JOINT_R_HipS      = join_conf('R_HipS',      coordinates(0.11,  0.89,  0.00))
JOINT_R_HipR      = join_conf('R_HipR',      coordinates(0.11,  0.73,  0.00))
JOINT_R_Knee      = join_conf('R_Knee',      coordinates(0.11,  0.49,  0.00))
JOINT_R_AnkleF    = join_conf('R_AnkleF',    coordinates(0.11,  0.10,  0.00))
JOINT_R_AnkleS    = join_conf('R_AnkleS',    coordinates(0.11,  0.10,  0.00))
#===========================================================================
# Joints_options
# (1)'motor_name',
# (2)'sticks_to',
# (3)'axes' X-Y-Z,
# (4)'min_angel',
# (5)'max_angel',
# (6)'angular_speed', 
# (7)'torque'
#                                1                 2                         3                 4   5   6   7
motor_optns = namedtuple('motor_optns',['motor_name', 'sticks_to', 'axes', 
                                          'min_angel','max_angel', 'angular_speed', 'torque'])
                              
MOTOR_HeadF           = motor_optns('HeadF',          'HeadR',          coordinates(0.0,-1.0, 0.0), -60, 20,130,450.0)
MOTOR_HeadR           = motor_optns('HeadR',          'Neck',           coordinates(0.0, 0.0,-1.0), -80, 80,420,172.0)
MOTOR_Neck            = motor_optns('Neck',           'TorsoR',         coordinates(0.0,-1.0, 0.0), -10, 40,110,360.0)
MOTOR_TorsoR          = motor_optns('TorsoR',         'TorsoF',         coordinates(0.0, 0.0,-1.0), -95, 95,230,525.0)
MOTOR_TorsoF          = motor_optns('TorsoF',         'TorsoS',         coordinates(0.0,-1.0, 0.0), -12, 40, 75,525.0)
MOTOR_TorsoS          = motor_optns('TorsoS',         'Pelvis',         coordinates(1.0, 0.0, 0.0), -15, 15,753, 75.0)
MOTOR_R_AnkleF        = motor_optns('R_AnkleF',       'R_Knee',         coordinates(0.0,-1.0, 0.0), -40, 25,180,240.0)
MOTOR_R_AnkleS        = motor_optns('R_AnkleS',       'R_AnkleF',       coordinates(1.0, 0.0, 0.0), -30, 30,210,150.0)
MOTOR_R_Knee          = motor_optns('R_Knee',         'R_HipR',         coordinates(0.0,-1.0, 0.0),   0,110,455,375.0)
MOTOR_R_ElbowR        = motor_optns('R_ElbowR',       'R_ShoulderS',    coordinates(0.0, 0.0,-1.0),-110,110,360,225.0)
MOTOR_R_ShoulderF     = motor_optns('R_ShoulderF',    'TorsoR',         coordinates(0.0,-1.0, 0.0),-150, 90,360,322.5)
MOTOR_R_Finger_Index  = motor_optns('R_Finger_Index', 'R_WristF',       coordinates(1.0, 0.0, 0.0),  12, 80, 90,  1.5)
MOTOR_R_ShoulderS     = motor_optns('R_ShoulderS',    'R_SHoulderF',    coordinates(1.0, 0.0, 0.0),-150,  0,180,495.0)
MOTOR_R_WristS        = motor_optns('R_WristS',       'R_WristR',       coordinates(1.0, 0.0, 0.0), -25, 15,100, 37.5)
MOTOR_R_WristF        = motor_optns('R_WristF',       'R_WristS',       coordinates(0.0,-1.0, 0.0), -18, 33,100, 37.5)
MOTOR_R_Finger_Middle = motor_optns('R_Finger_Middle','R_WristF',       coordinates(1.0, 0.0, 0.0),  12, 80, 90,  1.5)
MOTOR_R_HipS          = motor_optns('R_HipS',         'R_HipF',         coordinates(1.0, 0.0, 0.0), -90, 12, 70,345.0)
MOTOR_R_Finger_ThumbS = motor_optns('R_Finger_ThumbS','R_WristF',       coordinates(0.0, 0.0,-1.0), -90, 10, 90,  1.5)
MOTOR_R_Elbow         = motor_optns('R_Elbow',        'R_ElbowR',       coordinates(0.0,-1.0, 0.0),-120,  5,230,255.0)
MOTOR_R_HipR          = motor_optns('R_HipR',         'R_HipS',         coordinates(0.0, 0.0,-1.0), -90, 90,113,240.0)
MOTOR_R_HipF          = motor_optns('R_HipF',         'Pelvis',         coordinates(0.0,-1.0, 0.0), -90, 35,350,450.0)
MOTOR_R_Finger_Ring   = motor_optns('R_Finger_Ring',  'R_WristF',       coordinates(1.0, 0.0, 0.0),  12, 80, 90,  1.5)
MOTOR_R_Finger_Little = motor_optns('R_Finger_Little','R_WristF',       coordinates(1.0, 0.0, 0.0),  12, 80, 90,  1.5)
MOTOR_R_Finger_Thumb  = motor_optns('R_Finger_Thumb', 'R_Finger_ThumbS',coordinates(1.0, 0.0, 0.0), -75,-10, 90,  1.5)
MOTOR_R_WristR        = motor_optns('R_WristR',       'R_Elbow',        coordinates(0.0, 0.0,-1.0),-110,110,280, 75.0)
MOTOR_L_AnkleS        = motor_optns('L_AnkleS',       'L_AnkleF',       coordinates(1.0, 0.0, 0.0), -30, 30,210,150.0)
MOTOR_L_AnkleF        = motor_optns('L_AnkleF',       'L_Knee',         coordinates(0.0,-1.0, 0.0), -40, 25,180,240.0)
MOTOR_L_Knee          = motor_optns('L_Knee',         'L_HipR',         coordinates(0.0,-1.0, 0.0),   0,110,455,375.0)
MOTOR_L_ElbowR        = motor_optns('L_ElbowR',       'L_ShoulderS',    coordinates(0.0, 0.0,-1.0),-110,110,360,225.0)
MOTOR_L_ShoulderF     = motor_optns('L_ShoulderF',    'TorsoR',         coordinates(0.0,-1.0, 0.0),-150, 90,360,322.5)
MOTOR_L_Finger_Index  = motor_optns('L_Finger_Index', 'L_WristF',       coordinates(1.0, 0.0, 0.0), -80,-12, 90,  1.5)
MOTOR_L_ShoulderS     = motor_optns('L_ShoulderS',    'L_ShoulderF',    coordinates(1.0, 0.0, 0.0),   0,150,180,495.0)
MOTOR_L_WristS        = motor_optns('L_WristS',       'L_WristR',       coordinates(1.0, 0.0, 0.0), -15, 25,100, 37.5)
MOTOR_L_WristF        = motor_optns('L_WristF',       'L_WristS',       coordinates(0.0,-1.0, 0.0), -18, 33,100, 37.5)
MOTOR_L_HipS          = motor_optns('L_HipS',         'L_HipF',         coordinates(1.0, 0.0, 0.0), -12, 90, 70,345.0)
MOTOR_L_Finger_Middle = motor_optns('L_Finger_Middle','L_WristF',       coordinates(1.0, 0.0, 0.0), -80,-12, 90,  1.5)
MOTOR_L_Finger_ThumbS = motor_optns('L_Finger_ThumbS','L_WristF',       coordinates(0.0, 0.0,-1.0), -10, 90, 90,  1.5)
MOTOR_L_HipR          = motor_optns('L_HipR',         'L_HipS',         coordinates(0.0, 0.0,-1.0), -90, 90,113,240.0)
MOTOR_L_Elbow         = motor_optns('L_Elbow',        'L_ElbowR',       coordinates(0.0,-1.0, 0.0),-120,  5,230,255.0)
MOTOR_L_HipF          = motor_optns('L_HipF',         'Pelvis',         coordinates(0.0,-1.0, 0.0), -90, 35,350,450.0)
MOTOR_L_Finger_Ring   = motor_optns('L_Finger_Ring',  'L_WristF',       coordinates(1.0, 0.0, 0.0), -80,-12, 90,  1.5)
MOTOR_L_Finger_Little = motor_optns('L_Finger_Little','L_WristF',       coordinates(1.0, 0.0, 0.0), -80,-12, 90,  1.5)
MOTOR_L_Finger_Thumb  = motor_optns('L_Finger_Thumb', 'L_Finger_ThumbS',coordinates(1.0, 0.0, 0.0),  10, 75, 90,  1.5)
MOTOR_L_WristR        = motor_optns('L_WristR',       'L_Elbow',        coordinates(0.0, 0.0,-1.0),-110,110,280, 75.0)

Block           = namedtuple('Building_Block', ['DETAIL_CFG', 'JOINT_COORDINATE', 'MOTOR_CFG'])
HeadF           = Block(MOTOR_HeadF,           MOTOR_HeadF,           MOTOR_HeadF          )
HeadR           = Block(MOTOR_HeadR,           MOTOR_HeadR,           MOTOR_HeadR          )
Neck            = Block(MOTOR_Neck,            MOTOR_Neck,            MOTOR_Neck           )
TorsoR          = Block(MOTOR_TorsoR,          MOTOR_TorsoR,          MOTOR_TorsoR         )
TorsoF          = Block(MOTOR_TorsoF,          MOTOR_TorsoF,          MOTOR_TorsoF         )
TorsoS          = Block(MOTOR_TorsoS,          MOTOR_TorsoS,          MOTOR_TorsoS         )
R_AnkleF        = Block(MOTOR_R_AnkleF,        MOTOR_R_AnkleF,        MOTOR_R_AnkleF       )
R_AnkleS        = Block(MOTOR_R_AnkleS,        MOTOR_R_AnkleS,        MOTOR_R_AnkleS       )
R_Knee          = Block(MOTOR_R_Knee,          MOTOR_R_Knee,          MOTOR_R_Knee         )
R_ElbowR        = Block(MOTOR_R_ElbowR,        MOTOR_R_ElbowR,        MOTOR_R_ElbowR       )
R_ShoulderF     = Block(MOTOR_R_ShoulderF,     MOTOR_R_ShoulderF,     MOTOR_R_ShoulderF    )
R_Finger_Index  = Block(MOTOR_R_Finger_Index,  MOTOR_R_Finger_Index,  MOTOR_R_Finger_Index )
R_ShoulderS     = Block(MOTOR_R_ShoulderS,     MOTOR_R_ShoulderS,     MOTOR_R_ShoulderS    )
R_WristS        = Block(MOTOR_R_WristS,        MOTOR_R_WristS,        MOTOR_R_WristS       )
R_WristF        = Block(MOTOR_R_WristF,        MOTOR_R_WristF,        MOTOR_R_WristF       )
R_Finger_Middle = Block(MOTOR_R_Finger_Middle, MOTOR_R_Finger_Middle, MOTOR_R_Finger_Middle)
R_HipS          = Block(MOTOR_R_HipS,          MOTOR_R_HipS,          MOTOR_R_HipS         )
R_Finger_ThumbS = Block(MOTOR_R_Finger_ThumbS, MOTOR_R_Finger_ThumbS, MOTOR_R_Finger_ThumbS)
R_Elbow         = Block(MOTOR_R_Elbow,         MOTOR_R_Elbow,         MOTOR_R_Elbow        )
R_HipR          = Block(MOTOR_R_HipR,          MOTOR_R_HipR,          MOTOR_R_HipR         )
R_HipF          = Block(MOTOR_R_HipF,          MOTOR_R_HipF,          MOTOR_R_HipF         )
R_Finger_Ring   = Block(MOTOR_R_Finger_Ring,   MOTOR_R_Finger_Ring,   MOTOR_R_Finger_Ring  )
R_Finger_Little = Block(MOTOR_R_Finger_Little, MOTOR_R_Finger_Little, MOTOR_R_Finger_Little)
R_Finger_Thumb  = Block(MOTOR_R_Finger_Thumb,  MOTOR_R_Finger_Thumb,  MOTOR_R_Finger_Thumb )
R_WristR        = Block(MOTOR_R_WristR,        MOTOR_R_WristR,        MOTOR_R_WristR       )
L_AnkleS        = Block(MOTOR_L_AnkleS,        MOTOR_L_AnkleS,        MOTOR_L_AnkleS       )
L_AnkleF        = Block(MOTOR_L_AnkleF,        MOTOR_L_AnkleF,        MOTOR_L_AnkleF       )
L_Knee          = Block(MOTOR_L_Knee,          MOTOR_L_Knee,          MOTOR_L_Knee         )
L_ElbowR        = Block(MOTOR_L_ElbowR,        MOTOR_L_ElbowR,        MOTOR_L_ElbowR       )
L_ShoulderF     = Block(MOTOR_L_ShoulderF,     MOTOR_L_ShoulderF,     MOTOR_L_ShoulderF    )
L_Finger_Index  = Block(MOTOR_L_Finger_Index,  MOTOR_L_Finger_Index,  MOTOR_L_Finger_Index )
L_ShoulderS     = Block(MOTOR_L_ShoulderS,     MOTOR_L_ShoulderS,     MOTOR_L_ShoulderS    )
L_WristS        = Block(MOTOR_L_WristS,        MOTOR_L_WristS,        MOTOR_L_WristS       )
L_WristF        = Block(MOTOR_L_WristF,        MOTOR_L_WristF,        MOTOR_L_WristF       )
L_HipS          = Block(MOTOR_L_HipS,          MOTOR_L_HipS,          MOTOR_L_HipS         )
L_Finger_Middle = Block(MOTOR_L_Finger_Middle, MOTOR_L_Finger_Middle, MOTOR_L_Finger_Middle)
L_Finger_ThumbS = Block(MOTOR_L_Finger_ThumbS, MOTOR_L_Finger_ThumbS, MOTOR_L_Finger_ThumbS)
L_HipR          = Block(MOTOR_L_HipR,          MOTOR_L_HipR,          MOTOR_L_HipR         )
L_Elbow         = Block(MOTOR_L_Elbow,         MOTOR_L_Elbow,         MOTOR_L_Elbow        )
L_HipF          = Block(MOTOR_L_HipF,          MOTOR_L_HipF,          MOTOR_L_HipF         )
L_Finger_Ring   = Block(MOTOR_L_Finger_Ring,   MOTOR_L_Finger_Ring,   MOTOR_L_Finger_Ring  )
L_Finger_Little = Block(MOTOR_L_Finger_Little, MOTOR_L_Finger_Little, MOTOR_L_Finger_Little)
L_Finger_Thumb  = Block(MOTOR_L_Finger_Thumb,  MOTOR_L_Finger_Thumb,  MOTOR_L_Finger_Thumb )
L_WristR        = Block(MOTOR_L_WristR,        MOTOR_L_WristR,        MOTOR_L_WristR       )

def createMultiBody(baseMass, 
                    baseCollisionShapeIndex = None, 
                    baseVisualShapeIndex = None,
                    basePosition = None,
                    baseOrientation = None,
                    baseInertialFramePosition = None,
                    baseInertialFrameOrientation = None,
                    linkMasses = None,
                    linkCollisionShapeIndices = None,
                    linkVisualShapeIndices = None,
                    linkPositions = None,
                    linkOrientations = None,
                    linkInertialFramePositions = None,
                    linkInertialFrameOrientations = None,
                    linkParentIndices = None,
                    linkJointTypes = None,
                    linkJointAxis = None,
                    useMaximalCoordinates = None,
                    flags = None,
                    batchPositions = None,
                    physicsClientId = None
                   ):
    """
    
    """
    result = p.createMultiBody(
        baseMass                      = baseMass,
        baseCollisionShapeIndex       = baseCollisionShapeIndex,
        baseVisualShapeIndex          = baseVisualShapeIndex, 
        basePosition                  = basePosition,
        baseOrientation               = baseOrientation, 
        baseInertialFramePosition     = baseInertialFramePosition, 
        baseInertialFrameOrientation  = baseInertialFrameOrientation,
        linkMasses                    = linkMasses,
        linkCollisionShapeIndices     = linkCollisionShapeIndices,
        linkVisualShapeIndices        = linkVisualShapeIndices,
        linkPositions                 = linkPositions, 
        linkOrientations              = linkOrientations, 
        linkInertialFramePositions    = linkInertialFramePositions, 
        linkInertialFrameOrientations = linkInertialFrameOrientations, 
        linkParentIndices             = linkParentIndices,
        linkJointTypes                = linkJointTypes,
        linkJointAxis                 = linkJointAxis, 
        useMaximalCoordinates         = useMaximalCoordinates,   
        flags                         = flags, 
        batchPositions                = batchPositions,
        physicsClientId               = physicsClientId
    )
    return result