import pybullet as p
import _utils_v2 as utils
from _utils_v2 import xyz_v2, ROBOT, init_window

class main:
    def __init__(self, cfgs, body_Mass, centered = False):
        super(main, self).__init__()
        self.cfgs = cfgs.BUILD_BODY()
        self.body_Mass = body_Mass
        self.VARIABLE = 0.04
        self.f_start  = True
        self.centered = centered
        
    def create_body(self):
        '''
        0 Name
        1 CSI
        2 Masses
        3 Positions
        4 indices
        5 jntTypes
        6 axis
        7 ang_speed
        8 angles
        9 torque
        '''
        linkCSI       = [p[1] for p in self.cfgs]
        link_Masses   = [p[2] for p in self.cfgs]
        linkPositions = [p[3] for p in self.cfgs]
        indices       = [p[4] for p in self.cfgs]
        jointTypes    = [p[5] for p in self.cfgs]
        axis          = [p[6] for p in self.cfgs]
        ang_speed     = [p[7] for p in self.cfgs if p[7] is not None]
        angles        = [p[8] for p in self.cfgs if p[8] is not None] 
        torque        = [p[9] for p in self.cfgs if p[9] is not None] 
        
        # Torso     = [1,3,5] 
        # NandH     = [7,9,11] 
        # Right_arm = [13,15,17,19]
        # Left_arm  = [21,23,25,27]
        # Right_leg = [29,31,33,35,37,39]
        # Left_leg  = [42,44,46,48,50,52]
        # #BLOCK Init
        # top = Torso + NandH + Right_arm + Left_arm
        # bot = Right_leg + Left_leg
        # # default params 
        # motorIndexes = top + bot
        
        motorIndexes = [i for i, m in enumerate(self.cfgs) if m.angles is not None]
        minmax_angles= [m.angles for m in self.cfgs if m.angles is not None]
        linkValues   = [0] * len(motorIndexes)

        visualShapeId   = -1
        nlnk            = len(link_Masses)
        LVSIndices      = [-1] * nlnk 
        LOrientations   = [[0,0,0,1]] * nlnk
        LIFPositions    = [[0,0,0]]   * nlnk
        LIFOrientations = [[0,0,0,1]] * nlnk
        #Drop the body in the scene at the following body coordinates
        baseOrientation = [0,0,0,1]
        robot_positions = [[0,0,1]] # b_p.Z 
        coeff_P         = [10,10,0.1] if self.centered else [0.1,0.1,0.1]
        
        for basePosition in robot_positions:
            robot = p.createMultiBody(
                baseMass                      = self.body_Mass,
                baseCollisionShapeIndex       = utils.geom_shape(p.GEOM_BOX, 
                                                              height  = self.VARIABLE, 
                                                              coeff_P = coeff_P),
                baseVisualShapeIndex          = visualShapeId,
                basePosition                  = basePosition,
                baseOrientation               = baseOrientation,

                linkVisualShapeIndices        = LVSIndices,
                linkOrientations              = LOrientations,
                linkInertialFramePositions    = LIFPositions,
                linkInertialFrameOrientations = LIFOrientations,

                linkMasses                    = link_Masses,
                linkCollisionShapeIndices     = linkCSI,
                linkPositions                 = linkPositions,
                linkParentIndices             = indices,
                linkJointTypes                = jointTypes,
                linkJointAxis                 = axis
            )
            utils.resetCoordinates(robot, motorIndexes, linkValues, torque, ang_speed)
            

        # Add earth like gravity
        p.setGravity(0,0,-9.81)
        p.setRealTimeSimulation(1)

        self.cameraDistance = 3.0
        self.cameraYaw      = 211.20
        self.cameraPitch    = -46

        utils.resetCamera(
            cameraDistance = self.cameraDistance,
            cameraYaw      = self.cameraYaw,
            cameraPitch    = self.cameraPitch,
        )  
        return motorIndexes, minmax_angles