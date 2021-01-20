import pybullet as p
import time
import sys
import os
sys.path.append('../')
from .printer import printer
import _utils_v2 as u

def move_member(robot,motor_indexes, angles, minmax, member, joint, side = None):
    flag = True
    msg = 'Increase angle(UP), Decrease angle(DOWN), Quit(3)'
    while flag:
        if side is None:
            printer('Move {}_{}: Increase angle(UP), Decrease angle(DOWN), Reset0(B),Info(?), Quit(3)'.format(member,joint))
        else:
            printer('Move {} {}_{}: Increase angle(UP), Decrease angle(DOWN), Info(?), Quit(3)'.format(side,member,joint))
        key = p.getKeyboardEvents()
        if key.get(65297): 
            if angles[joint]<=minmax[1]:
                angles[joint]+=1
        if key.get(65298): 
            if angles[joint]>minmax[0]:
                angles[joint]-=1
        if key.get(98):    angles[joint] =0
        if key.get(47):    printer(msg); time.sleep(2)
        if key.get(51):    flag = False
        u.resetCoordinates(robot,motor_indexes,angles)
        time.sleep(0.005)
    return angles[joint]

def move_head(robot, motor_indexes, angles, minmax):
    flag = True
    u.resetCamera(
        cameraDistance = 2.60,
        cameraYaw      = 229.60,
        cameraPitch    = -61.60,
    )
    msg = 'MV Head Block: press "2" to go "MV Block"'
    while flag:
        printer('Move Head: Neck(N), HeadR(R), HeadF(F), Info(?), Quit(2)')
        key = p.getKeyboardEvents()
        if key.get(110): angles[0] = move_member(robot,motor_indexes, angles, minmax[0],'Neck', 0) # move_Neck(angle_neck)
        if key.get(114): angles[1] = move_member(robot,motor_indexes, angles, minmax[1],'HeadR', 1) # move_HeadR(angleRHead)
        if key.get(102): angles[2] = move_member(robot,motor_indexes, angles, minmax[2],'HeadF', 2) # move_HeadF(angleFHead)
        if key.get(47):  printer(msg)
        if key.get(50):  flag = False
        time.sleep(0.3)
    return angles

def move_torso(robot, motor_indexes, angles, minmax):
    flag = True
    msg = 'MV Torso Block: press "2" to go "MV Block"'
    joints = ['TorsoS', 'TorsoF', 'TorsoR']
    while flag:
        printer('Move Torso: TorsoS(S), TorsoF(F), TorsoR(R), Info(?), Quit(2)')
        key = p.getKeyboardEvents()
        if key.get(115): angles[0] = move_member(robot,motor_indexes,angles,minmax[0],joints[0], 0) # move_TorsoS(angle_TorsoS)
        if key.get(102): angles[1] = move_member(robot,motor_indexes,angles,minmax[1],joints[1], 1) # move_TorsoF(angle_TorsoF)
        if key.get(114): angles[2] = move_member(robot,motor_indexes,angles,minmax[2],joints[2], 2) # move_TorsoR(angle_TorsoR)
        if key.get(47):  printer(msg)
        if key.get(50):  flag = False
        time.sleep(0.3)
    return angles

def Side(robot, motor_indexes, angles, minmax, side, ops, joints):
    RL = 0 if side == 'Right' else 1 # Right OR Left
    indexes = [i for i in range(len(motor_indexes)) if i%2==RL]
    flag = True
    msg  = 'MV Arms Block: press "Q" to go "MV Block"'
    while flag:
        printer(ops.format(side))
        key = p.getKeyboardEvents()
        if key.get(102):angles[indexes[0]] = move_member(robot,motor_indexes,angles,minmax[0],    joints[0],indexes[0],side)
        if key.get(115):angles[indexes[1]] = move_member(robot,motor_indexes,angles,minmax[1][RL],joints[1],indexes[1],side) 
        if key.get(114):angles[indexes[2]] = move_member(robot,motor_indexes,angles,minmax[2],    joints[2],indexes[2],side)
        if key.get(101):angles[indexes[3]] = move_member(robot,motor_indexes,angles,minmax[3],    joints[3],indexes[3],side)
        if len(motor_indexes)>7:
            if key.get(116):angles[indexes[3]] = move_member(robot,motor_indexes,angles,minmax[4],joints[4],indexes[4],side)
            if key.get(121):angles[indexes[3]] = move_member(robot,motor_indexes,angles,minmax[5],joints[5],indexes[5],side)
        if key.get(47): printer(msg); time.sleep(2)
        if key.get(113):flag = False
        time.sleep(0.3)
                
    return angles

def move_arms(robot, motor_indexes, angles, minmax):
    flag = True
    msg = 'MV Arms Block: press "2" to go "MV Block"'
    ops = 'Move {} Arm: ShoulderF(F),ShoulderF(S), ElbowR(R), Elbow(E), Info(?), Quit(Q)'
    joints = ['ShoulderF', 'ShoulderS','ElbowR','Elbow']
    while flag:
        printer('Move Arms: Left(L), Right(R), Info(?), Quit(2)')
        key = p.getKeyboardEvents()
        if key.get(108): angles = Side(robot, motor_indexes, angles, minmax, 'Left',  ops, joints)# Left Side 
        if key.get(114): angles = Side(robot, motor_indexes, angles, minmax, 'Right', ops, joints)# Right Side
        if key.get(47):  printer(msg)
        if key.get(50):  flag = False
        time.sleep(0.3)
    return angles

def move_legs(robot, motor_indexes, angles, minmax):
    flag = True
    msg = 'MV Legs Block: press "2" to go "MV Block"'
    ops = 'Move {} Arm: HipF(F), HipS(S), HipR(R), Knee(E), AnkleF(T), AnkleS(Y), Info(?), Quit(Q)'
    joints = ['HipF','HipS','HipR','Knee','AnkleF', 'AnkleS']
    while flag:
        printer('Move Legs: Left(L), Right(R), Info(?), Quit(2)')
        key = p.getKeyboardEvents()
        if key.get(108): angles = Side(robot, motor_indexes, angles, minmax, 'Left',  ops, joints)# Left Side
        if key.get(114): angles = Side(robot, motor_indexes, angles, minmax, 'Right', ops, joints)# Right Side
        if key.get(47):  printer(msg)
        if key.get(50):  flag = False
        time.sleep(0.3)
    return angles