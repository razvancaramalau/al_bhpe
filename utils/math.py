import numpy as np
from math import pi
_EPSILON = 10e-8

def sigmoid(x):
    return  1.0 / (1 + np.exp(-x+_EPSILON))

def get_angle_between_two_lines(line0,line1=(0,-1)):
    rot =np.arccos(np.dot(line0,line1)/np.linalg.norm(line0,axis=1))
    loc_neg = np.where(line0[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = np.cast['float32'](rot/pi*180)
    return rot