import pybullet as p
import time
import sys
import os
sys.path.append('../')
from .printer import printer
import _utils_v2 as u

def options(cameraDistance, cameraYaw, cameraPitch):
    msg = 'OPTIONS: press "1" to go "Main"'
    printer(msg)
    flag = True
    while flag:
        print_dict = {
            'cameraDistance':round(cameraDistance,4),
            'cameraYaw':cameraYaw,
            'cameraPitch':cameraPitch}
        printer(print_dict)
        options = p.getKeyboardEvents()
        if options.get(65298):# DOWN
            cameraDistance+=0.1; printer("cameraDistance '+'")
        if options.get(65297):# UP
            cameraDistance-=0.1; printer("cameraDistance '-'")
        if options.get(65296):#
            cameraYaw+=1; printer("cameraYaw '+'")
        if options.get(65295):#
            cameraYaw-=1; printer("cameraYaw '+'")
        if options.get(47):# HELP
            printer(msg)
            time.sleep(2)
        if options.get(49):
            printer('QUIT OPTIONS')
            flag = False
        u.resetCamera(cameraDistance = cameraDistance,
                      cameraYaw      = cameraYaw,
                      cameraPitch    = cameraPitch)
        time.sleep(0.025)
    return cameraDistance, cameraYaw, cameraPitch