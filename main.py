import h5py
import keras
import numpy as np
import os
import tensorflow as tf
import glob 
import sys
import gc

from argparse import ArgumentParser
from load_data import load_data_dp
from train_test import train_all_dp, train_dp_orig
from config import ADDIM, MC_DROPOUT, NO_EPOCHS, BATCH_SIZE, LR, GPUv 

parser = ArgumentParser(description="Code for running hand detection on training and validation.")
parser.add_argument("-d", "--dataset", type = str, default='icvl', # mega/nyu
                                                    help='Dataset name (default: %(default)s)')
parser.add_argument("-m", "--method", type = str, default="cke", #kcg
                                                    help='Active learning selection method (default: %(default)s)')
parser.add_argument("-c", "--MC", type = bool, default=False, #kcg
                                                    help='Bayesian adaptation active(default: %(default)r)')    
parser.add_argument("-v", "--MC_VAR", type = int, default=0, #kcg
                                                    help='Monte Carlo Variant 0-A, 1-B, 2-C (default: %(default)d)')         
parser.add_argument("-t", "--no_of_trials", type = int, default=5, 
                                                    help='Number of trials for AL (default: %(default)d)')      
parser.add_argument("-s", "--save_dir", type = str, default="data", 
                                                    help='Folder to save models and load datasets:  %(default)s)')     
parser.add_argument("-f", "--full", type = bool, default=True, #kcg
                                                    help='Use the entire dataset(default: %(default)b it required full \
                                                          pre-processed dataset)')                                                                                                             
args, unparsed = parser.parse_known_args()
# from al_utils import random_sampled_data_ex2

_EPSILON = 10e-8
os.environ["CUDA_VISIBLE_DEVICES"]= GPUv
save_dir = args.save_dir

setname = args.dataset
method  = args.method
MC = args.MC
RUNS = args.no_of_trials
entire_dataset = args.full

# Main
if __name__ == '__main__':
    for runs in range(0,RUNS,1):
        #runs = args.MC_VAR
        version = 'deepprior_%s%d_%s%d_lr%f'%(setname, runs, method, MC, LR)
        fid = open("results_deepprior_%s_%s%d_run%s.txt"%(setname, method, MC, str(runs)),"w")
        chunk_steps = [2,3,4,5,6,7,8,9,10]
        # Name of datasets:
        if setname == 'nyu':
            if entire_dataset:
                in_dataset = "nyu_train_128"
            else:
                in_dataset = "nyu_subset_"
            test_set = "nyu_test_128"
        elif setname == 'mega':
            if entire_dataset:
                in_dataset = "train_Hand128"
            else:
                in_dataset = "bh_subset_"
            test_set = "test_bigHand128"
        elif setname == 'icvl':
            if entire_dataset:
                in_dataset = "icvl_train_128"
            else:
                in_dataset = "icvl_subset_"
            test_set = "icvl_test_128"
        else:
            exit


        # Load testing data
        joint_no = np.arange((21))
        palm_idx =[0,1,5,9,13,17]
        
        # Load the testing data
        test_img0, test_target, _, _= load_data_dp(save_dir, test_set, joint_no)

        # selection step
        prev_train, prev_valid = np.zeros((ADDIM)), np.zeros((ADDIM))

        # train with random sampling with the initial 10% random selected data
        pool_stage = 1
        k = 0
        partA = [] # dummy
        # train hand on initial set
        if not MC:
            error_test,error_train,versionfull, prev_train, prev_valid  = train_dp_orig(save_dir, 'o' + version, in_dataset, test_set, 
                                                                                        partA, test_img0, test_target, k, pool_stage, 
                                                                                        MC_DROPOUT, NO_EPOCHS, BATCH_SIZE, LR, prev_train, prev_valid, 
                                                                                        method, palm_idx, MC, setname, entire_dataset)


        else:
            error_test,error_train,versionfull, prev_train, prev_valid = train_all_dp(save_dir, version, in_dataset, test_set, 
                                                                                    partA, test_img0, test_target, k, pool_stage, 
                                                                                    MC_DROPOUT, NO_EPOCHS, BATCH_SIZE, LR, prev_train, prev_valid, 
                                                                                    method, palm_idx, MC, setname, entire_dataset, args.MC_VAR)
        # write the results of the joint errors 
        fid.write(str(pool_stage)+" "+str(k)+" ")
        error_test.tofile(fid,sep=" ")
        fid.write(' ')
        np.mean(error_test).tofile(fid,sep=" ")
        fid.write('\n')
        fid.write(str(pool_stage)+" "+str(k)+" ")
        error_train.tofile(fid,sep=" ")
        fid.write(' ')
        np.mean(error_train).tofile(fid,sep=" ")
        fid.write('\n')

        if entire_dataset:
            sys.exit()
        # move to the selection of the unlabeled pool of data

        for idx,pool_stage in enumerate(chunk_steps):

            if idx == 0:

                selection_functions = []    
                selection_functions.append(partA)


            else:
                # Get indices of new data from acquisition functions
                pool_percentage = ADDIM
                selection_functions = []
                selection_functions.append(partA)



            prev_versions = []
            for k,partA in enumerate(selection_functions):
                if not MC:
                    error_test,error_train,versionfull, prev_train, prev_valid = train_dp_orig(save_dir, 'o' + version, in_dataset, test_set, 
                                                                                               partA, test_img0, test_target, k, pool_stage, 
                                                                                               MC_DROPOUT, NO_EPOCHS, BATCH_SIZE, LR, prev_train, prev_valid, 
                                                                                               method, palm_idx, MC, setname, entire_dataset)
                else:
                    error_test,error_train,versionfull, prev_train, prev_valid = train_all_dp(save_dir, version, in_dataset, test_set, 
                                                                                              partA, test_img0, test_target, k, pool_stage, 
                                                                                                MC_DROPOUT, NO_EPOCHS, BATCH_SIZE, LR, prev_train, prev_valid, 
                                                                                                method, palm_idx, MC, setname, entire_dataset, args.MC_VAR)
                prev_versions.append(versionfull)

                # write the results of the joint errors 
                fid.write(str(pool_stage)+" "+str(k)+" ")
                error_test.tofile(fid,sep=" ")
                fid.write(' ')
                np.mean(error_test).tofile(fid,sep=" ")
                fid.write('\n')
                fid.write(str(pool_stage)+" "+str(k)+" ")
                error_train.tofile(fid,sep=" ")
                fid.write(' ')
                np.mean(error_train).tofile(fid,sep=" ")
                fid.write('\n')



        fid.close()