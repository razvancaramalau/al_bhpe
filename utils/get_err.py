import h5py
import sys
sys.path.insert(0,"utils/")
import xyz_uvd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import math


def get_normuvd_from_offset(offset,pred_palm,jnt_uvd_in_prev_layer,scale):

    rot_angle = math.get_angle_between_two_lines(line0=(pred_palm[:,3,:]-pred_palm[:,0,:])[:,0:2])
    for i in range(offset.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),-rot_angle[i],1/scale)
        for j in range(offset.shape[1]):
            offset[i,j,0:2] = (np.dot(M,np.array([offset[i,j,0]*96+48,offset[i,j,1]*96+48,1]))-48)/96

    pred_uvd = jnt_uvd_in_prev_layer+offset
    return pred_uvd

def get_err_from_normuvd(path,normuvd,dataset,jnt_idx,setname):
    if setname =='icvl':
        centerU=320/2
    if setname =='nyu':
        centerU=640/2
    if setname =='msrc':
        centerU=512/2
    if setname=='mega':
        centerU=315.944855
    #print(path)
    f = h5py.File(path, 'r')
    xyz_gt=np.array([d for d in f['xyz_gt'][...] if sum(sum(d))!=0])
    
    if setname=='icvl' or setname=='nyu':
        # t = f['uvd_hand_centre'][...]
        uvd_hand_centre=np.array([d for d in f['uvd_hand_centre'][...] if sum(sum(d))!=0])
    else:
        uvd_hand_centre=np.array([d for d in f['uvd_hand_centre'][...] if sum(d)!=0])
    # bbsize=f['bbsize'][...]
    if setname=='icvl' or setname=='nyu':
        bbsize = f['bbsize'].value
    else:
        bbsize=f['bbsize'][0]
    f.close()
    if setname!='icvl'and setname!='nyu':
        uvd_hand_centre=np.expand_dims(uvd_hand_centre,axis=1)
    numImg=uvd_hand_centre.shape[0]

    bbsize_array = np.ones((numImg,3))*bbsize
    print(uvd_hand_centre.shape)
    bbsize_array[:,2]=uvd_hand_centre[:,0,2]
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bbsize_array)
    normUVSize = np.array(np.ceil(bbox_uvd[:,0]) - centerU,dtype='int32')
    normuvd=normuvd[:numImg].reshape(numImg,len(jnt_idx),3)
    uvd = np.empty_like(normuvd)
    uvd[:,:,2]=normuvd[:,:,2]*bbsize
    uvd[:,:,0:2]=normuvd[:,:,0:2]*normUVSize.reshape(numImg,1,1)
    uvd += uvd_hand_centre

    xyz_pred = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)

    err = np.mean(np.sqrt(np.sum((xyz_pred-xyz_gt[:,jnt_idx,:])**2,axis=-1)),axis=0)

    print(dataset,'err', err,np.mean(err))

    return xyz_pred,xyz_gt, err