import sys
import os
sys.path.append('../')
import _utils_v2 as u

def clamp(angles, minmax):
    assert len(angles) == len(minmax)
    for i in range(len(angles)):
        if angles[i] < minmax[i][0]:
            angles[i] = minmax[i][0]
            
        if angles[i] > minmax[i][1]:
            angles[i] = minmax[i][1]
    return angles

def move(cfgs, angles=None):
    l = [line.indices for line in cfgs if line.jntTypes==0]
    if angles is None:
        angles = [0]*len(l)
    
    minmax = [TorsoS,TorsoF,TorsoR]
    
    angles = clamp(angles, minmax)
    
    u.resetCoordinates(robot, motor_indexes, angles)
    
    reward, loss = 0,0
    return array, reward, loss 

# from .MAIN_LOOP.printer import printer
# from .MAIN_LOOP.options import options
# from .MAIN_LOOP.body_mv import move_head, move_torso, move_arms, move_legs, move
# 
# # angleFHead, angleRHead, angle_neck
# main_flag = True
# while main_flag:
#     msg = 'Main: Options(O), BodyMove(M), Info(?), Quit(Q)'
#     printer(msg)
#     keys = p.getKeyboardEvents()
#     if keys.get(111): # o - options
#         result = options(
#             body.cameraDistance, 
#             body.cameraYaw, 
#             body.cameraPitch)
#         body.cameraDistance, \
#         body.cameraYaw, \
#         body.cameraPitch = result
    
#     if keys.get(109): # m - move body paths
#         move_flag = True
#         while move_flag:
#             printer('Move: Head(H), Torso(T), Arms(A), Legs(L), Info(?). Quit(1)')
#             move_loop = p.getKeyboardEvents()
#             if move_loop.get(104):# H - Head
#                 angles      = [angle_neck, angle_HeadR, angle_HeadF]
#                 minmaxNandH = [Neck, HeadR, HeadF]
#                 angles      = move_head(robot, NandH, angles, minmaxNandH)
#             if move_loop.get(116):# T - Torso
#                 angles      = [angle_TorsoS, angle_TorsoF, angle_TorsoR]
#                 minmaxTorso = [TorsoS,TorsoF,TorsoR]
#                 angles      = move_torso(robot, Torso, angles, minmaxTorso)
#             if move_loop.get(97): # A - Arms
#                 angles      = [
#                     angle_R_ShoulderF, angle_L_ShoulderF,
#                     angle_R_ShoulderS, angle_L_ShoulderS,
#                     angle_R_ElbowR,    angle_L_ElbowR,
#                     angle_R_Elbow,     angle_L_Elbow,]
#                 minmaxArms  = [ShoulderF, [R_ShoulderS, L_ShoulderS], ElbwR, Elbw]
#                 Arms        = ShldrF + ShldrS + ElbowR + Elbow
#                 angles      = move_arms(robot, Arms, angles, minmaxArms)
#             if move_loop.get(108):# L - Legs
#                 angles      = [
#                     angle_R_HipF,   angle_L_HipF,
#                     angle_R_HipS,   angle_L_HipS,
#                     angle_R_HipR,   angle_L_HipR,
#                     angle_R_Knee,   angle_L_Knee,
#                     angle_R_AnkleF, angle_L_AnkleF,
#                     angle_R_AnkleS, angle_L_AnkleS,]
#                 minmaxLegs = [__HipF, [R_HipS, L_HipS], __HipR, __Knee, __AnkleF, __AnkleS]
#                 Legs       = HipF + HipS + HipR + Knee + AnkleF + AnkleS
#                 angles     = move_legs(robot, Legs, angles, minmaxLegs)
#             if move_loop.get(49): # Quit MV Block
#                 move_flag = False
#             if move_loop.get(47):
#                 msg = 'MV Block: press "1" to go "Main"'
#                 printer(msg)
#             time.sleep(0.3)
#     if keys.get(47):
#         printer(msg)
#     if keys.get(113):
#         printer('Quit Main...')
#         main_flag = False
#     time.sleep(0.1)
# p.disconnect()



angle_HeadF       = 0; angle_HeadR       = 0; angle_neck   = 0;
angle_TorsoS      = 0; angle_TorsoF      = 0; angle_TorsoR = 0;
angle_R_ShoulderF = 0; angle_L_ShoulderF = 0
angle_R_ShoulderS = 0; angle_L_ShoulderS = 0
angle_R_ElbowR    = 0; angle_L_ElbowR    = 0
angle_R_Elbow     = 0; angle_L_Elbow     = 0
angle_R_HipF      = 0; angle_L_HipF      = 0
angle_R_HipS      = 0; angle_L_HipS      = 0
angle_R_HipR      = 0; angle_L_HipR      = 0
angle_R_Knee      = 0; angle_L_Knee      = 0
angle_R_AnkleF    = 0; angle_L_AnkleF    = 0
angle_R_AnkleS    = 0; angle_L_AnkleS    = 0