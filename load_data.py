import h5py
import numpy as np

def load_data_dp(save_dir,dataset, joint_no):
    
    f = h5py.File('%s/source/%s.h5'%(save_dir,dataset), 'r')
    test_y= np.array([d for d in f['uvd_norm_gt'][...] if sum(sum(d))!=0]).reshape(-1,len(joint_no)*3)

    if dataset.startswith("test") or dataset.startswith("train"):
        test_x0 = np.array([d for d in f['img0'][...] if sum(sum(d))!=0])

        file_names = f['new_file_names'][...]
        meanUVD = f['uvd_hand_centre'][...]
    else:
        test_x0 = np.array([d for d in f['img'][...] if sum(sum(d))!=0])

        file_names = f['file_names'][...]
        meanUVD = f['uvd_hand_centre'][...]

    f.close()
    print(dataset,' loaded') #,test_x0.shape,test_y.shape)
    return np.expand_dims(test_x0,axis=-1), test_y, file_names, meanUVD