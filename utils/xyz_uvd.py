import numpy as np
from matplotlib import pylab
def convert_depth_to_uvd(depth):
    v, u = pylab.meshgrid(range(0, depth.shape[0], 1), range(0, depth.shape[1], 1), indexing= 'ij')
    v = np.asarray(v, 'uint16')[:, :, np.newaxis]
    u = np.asarray(u, 'uint16')[:, :, np.newaxis]
    depth = depth[:, :, np.newaxis]
    uvd = np.concatenate((u, v, depth), axis=2)

    # print v.shape,u.shape,uvd.shape
    return uvd


def xyz2uvd(setname,xyz):
    if setname=='mega':
        focal_length_x = 475.065948
        focal_length_y = 475.065857
        u0= 315.944855
        v0= 245.287079

        uvd = np.empty_like(xyz)
        if len(uvd.shape)==3:
            trans_x= xyz[:,:,0]
            trans_y= xyz[:,:,1]
            trans_z = xyz[:,:,2]
            uvd[:,:,0] = u0 + focal_length_x * ( trans_x / trans_z )
            uvd[:,:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
            uvd[:,:,2] = trans_z #convert m to mm
        else:
            trans_x= xyz[:,0]
            trans_y= xyz[:,1]
            trans_z = xyz[:,2]
            uvd[:,0] = u0 +  focal_length_x * ( trans_x / trans_z )
            uvd[:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
            uvd[:,2] = trans_z #convert m to mm
        return uvd




    if setname =='msrc' or setname =='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
        uvd = np.empty_like(xyz,dtype='float32')
        if len(xyz.shape)==3:
            trans_x= xyz[:,:,0]
            trans_y= xyz[:,:,1]
            trans_z = xyz[:,:,2]
            uvd[:,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[:,:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
            uvd[:,:,2] = trans_z #convert m to mm        
        else:
            trans_x= xyz[:,0]
            trans_y= xyz[:,1]
            trans_z = xyz[:,2]
            uvd[:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
            uvd[:,2] = trans_z #convert m to mm  

    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240
        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
        uvd = np.empty_like(xyz,dtype='float32')
        if len(xyz.shape)==3:
            trans_x= xyz[:,:,0]
            trans_y= xyz[:,:,1]
            trans_z = xyz[:,:,2]
            uvd[:,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[:,:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
            uvd[:,:,2] = trans_z #convert m to mm        
        else:
            trans_x= xyz[:,0]
            trans_y= xyz[:,1]
            trans_z = xyz[:,2]
            uvd[:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
            uvd[:,2] = trans_z #convert m to mm  
    if setname =='nyu'or setname =='NYU':
        res_x = 640
        res_y = 480
        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor

        #res_x = 320
        #res_y = 240
        cf_x = 588.036865
        cf_y = 587.075073
        uvd = np.empty_like(xyz,dtype='float32')
        if len(xyz.shape)==3:
            trans_x= xyz[:,:,0]
            trans_y= xyz[:,:,1]
            trans_z = xyz[:,:,2]
            uvd[:,:,2] = trans_z #convert m to mm        
            uvd[:,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[:,:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
        else:
            trans_x= xyz[:,0]
            trans_y= xyz[:,1]
            trans_z = xyz[:,2]
            uvd[:,2] = trans_z #convert m to mm
            uvd[:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )

    return uvd

def uvd2xyz(setname,uvd):
    if setname =='mega':
        focal_length_x = 475.065948
        focal_length_y = 475.065857
        u0= 315.944855
        v0= 245.287079
        xyz = np.empty_like(uvd)
        if len(uvd.shape)==3:

            xyz[:,:,2] = uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - u0)/focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- v0)/focal_length_y*xyz[:,:,2]
        else:

            z =  uvd[:,2] # convert mm to m
            xyz[:,2]=z
            xyz[:,0] = ( uvd[:,0]- u0)/focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- v0)/focal_length_y*z
        return xyz

    if setname =='msrc' or setname=='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
        if len(uvd.shape)==3:
            xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
            xyz[:,:,2]=uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
        else:
            xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
            z =  uvd[:,2] # convert mm to m
            xyz[:,2]=z
            xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z   
    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
        if len(uvd.shape)==3:
            xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
            xyz[:,:,2]=uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
        else:
            xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
            z =  uvd[:,2] # convert mm to m
            xyz[:,2]=z
            xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z   
    if setname =='nyu' or setname=='NYU':
        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor

        if len(uvd.shape)==3:
            xyz = np.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
            xyz[:,:,2]=uvd[:,:,2]
            xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2.0)/res_x/ focal_length_x*xyz[:,:,2]
            xyz[:,:,1] = ( uvd[:,:,1]- res_y / 2.0)/res_y/focal_length_y*xyz[:,:,2]
        else:
            xyz = np.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
            xyz[:,2] = uvd[:,2]
            xyz[:,0] = ( uvd[:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[:,1] = ( uvd[:,1]- res_y / 2)/res_y/focal_length_y*z
 
    return xyz
