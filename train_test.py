import os
import numpy as np
import tensorflow as tf
import keras
import time
from keras import metrics
from keras.optimizers import Adam
from keras.models import model_from_json
from sklearn.utils import shuffle
from scipy.stats import entropy

from load_data import load_data_dp
from models import hpe
from utils.kcenterGreedy import kCenterGreedy, kCenterGreedyUnc4
from utils.losses import bayesian_loss_with_uncertainty, mean_squared_errordp
from utils import get_err
from config import GPUv, ADDIM

def predict_features_dp(save_dir, version, in_dataset, palm_idx, prev_train, pool_stage, lr, MC, MC_dropout, setname):

    dims = 63
    monte_carlo_simulations = MC_dropout

    version_model = '_'.join(version.split('_')[:-2])
    # load weights into new model
    with tf.device("/gpu:"+GPUv):
        json_file = open("%s/deepprior/%s.json"%(save_dir,version_model), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        if MC:
            loaded_model.load_weights("%s/deepprior/weight_mc_%s"%(save_dir,version))
        else:
            loaded_model.load_weights("%s/deepprior/weight_%s"%(save_dir,version))

    model = loaded_model
    # for dataset in in_dataset:
    test_x0,_,_,_ = load_data_dp(save_dir, in_dataset + str(pool_stage), palm_idx)
    test_x0 = np.concatenate((test_x0, prev_train))

    if MC:
        features_mean = np.zeros([monte_carlo_simulations,test_x0.shape[0],2*63]) #7500])
    else:
        features_mean = np.zeros([test_x0.shape[0],63])#7500])

    if MC:
        for t in range(monte_carlo_simulations):
            features_mean[t] = model.predict(x={'input0':test_x0},batch_size=128)[0]

        epistemics_var = np.std(features_mean, axis=0)[:,0:dims]
        features_mean = np.mean(features_mean, axis=0)
        features_mean = np.array(features_mean[:,0:dims])
    else:
        features_mean = model.predict(x={'input0':test_x0},batch_size=128)
        features_mean = features_mean.reshape(-1, 63)
        epistemics_var = np.zeros_like(features_mean) #dummy

    del test_x0
    return features_mean, epistemics_var


def geom_kcg(save_dir, dataset, version, pool_size, pool_stage, pool_percentage, prev_train, lr, palm_idx, MC, MC_dropout, setname):
    start_time = time.time()
    features, _ = predict_features_dp(save_dir, version, dataset, palm_idx, prev_train, pool_stage, lr, MC, MC_dropout, setname)
    print("Feature processed in %s" %(time.time() - start_time))
    sampling = kCenterGreedy(features)
    del features
    av_idx = np.arange(pool_size,pool_size+(pool_stage-1)*ADDIM)
    batch = sampling.select_batch_(av_idx, pool_percentage)
    print("kcg locations find in %s" %(time.time() - start_time))
    del av_idx
    return batch

def geom_cke(save_dir, dataset, version, pool_size, pool_stage, pool_percentage, prev_train, lr, palm_idx, MC, MC_dropout, setname):

    start_time = time.time()
    features, uncertainties = predict_features_dp(save_dir, version, dataset, palm_idx, prev_train, pool_stage, lr, MC, MC_dropout, setname)
    print("Feature processed in %s" %(time.time() - start_time))
    sampling = kCenterGreedyUnc4(features)
    del features
    av_idx = np.arange(pool_size,pool_size+(pool_stage-1)*ADDIM)
    batch = sampling.select_batch_unc4_(av_idx, pool_percentage, np.expand_dims(uncertainties, axis=1), 0.3)
    print("cke locations find in %s" %(time.time() - start_time))
    del av_idx

    return batch

def predict_dp(save_dir, version, dataset, palm_idx, lr, MC_dropout, setname):
    
    version_model = '_'.join(version.split('_')[:-2])

    # load weights into new model
    with tf.device("/gpu:"+GPUv):
        json_file = open("%s/deepprior/%s.json"%(save_dir,version_model), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("%s/deepprior/weight_%s"%(save_dir,version))

        loaded_model.compile(optimizer=Adam(lr=lr),
                    loss={"dense_3":mean_squared_errordp},metrics={'dense_3': metrics.mean_squared_error})

    test_x0,_,_,_ = load_data_dp(save_dir, dataset, palm_idx)

    with tf.device("/gpu:"+GPUv):
        normuvd = loaded_model.predict(x={'input0':test_x0},batch_size=128)

    normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
    np.save("%s/deepprior/results/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)
    normuvd = np.load("%s/deepprior/results/%s_normuvd_%s.npy"%(save_dir, dataset, version)) # change to normuvd without %s_                                                                                 
    path='%s/source/%s.h5'%(save_dir,dataset)
    _, _, xyz_err = get_err.get_err_from_normuvd(path=path,dataset=dataset,normuvd=normuvd[:,:,:],jnt_idx=palm_idx,setname=setname)

    return xyz_err


def predict_with_aleatoric_dp(save_dir, version, dataset, palm_idx, lr, MC_dropout, setname):
    data_augmentation = False
    dims = 63
    aleatoric_monte_carlo_simulations = 100
    version_model = '_'.join(version.split('_')[:-2])

    # load weights into new model
    with tf.device("/gpu:"+GPUv):
        json_file = open("%s/deepprior/%s.json"%(save_dir,version_model), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("%s/deepprior/weight_mc_%s"%(save_dir,version))

        loaded_model.compile(optimizer=Adam(lr=lr),
                        loss={"regress_variance": bayesian_loss_with_uncertainty(aleatoric_monte_carlo_simulations, dims),
                                "dense_3":mean_squared_errordp},metrics={'dense_3': metrics.mean_squared_error},
                                loss_weights={'regress_variance': 1., 'dense_3': 1.})

    epistemic_MC_dropout = MC_dropout
    #for dataset in in_dataset:
    test_x0,_,_,_ = load_data_dp(save_dir,dataset, palm_idx)
    # start=time.clock()
    # valid_lost = np.ones((test_x0.shape[0],dims))#*best_lost
    
    normuvd_mean = np.zeros([epistemic_MC_dropout,test_x0.shape[0],2*dims])

    if data_augmentation:
        normuvd_mean_aug = np.zeros([epistemic_MC_dropout,test_x0.shape[0],2*dims])
        loc = 1
        if setname == "nyu":
            loc = 0.85
    for t in range(epistemic_MC_dropout):
        with tf.device("/gpu:"+GPUv):
            normuvd_mean[t] = loaded_model.predict(x={'input0':test_x0},batch_size=128)[0]
    # print(normuvd[0])
    del test_x0
    epistemic_variances = np.var(normuvd_mean,axis=0)[:,0:dims]

    normuvd_ext = np.mean(normuvd_mean, axis=0)
    del normuvd_mean
    normuvd = np.array(normuvd_ext[:,0:dims])
    aleatoric_variances = np.array(normuvd_ext[:,dims:])
    # print('fps',test_x0.shape[0]/(time.clock()-start))
    normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
    aleatoric_variances.shape=(aleatoric_variances.shape[0],len(palm_idx),3)
    epistemic_variances.shape=(epistemic_variances.shape[0],len(palm_idx),3)
    np.save("%s/deepprior/results/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)
    np.save("%s/deepprior/results/%s_aleatoric_var_%s"%(save_dir,dataset,version),aleatoric_variances)
    np.save("%s/deepprior/results/%s_epistemic_var_%s"%(save_dir,dataset,version),epistemic_variances)
    normuvd = np.load("%s/deepprior/results/%s_normuvd_%s.npy"%(save_dir, dataset, version)) # change to normuvd without %s_                                                                                 
    path='%s/source/%s.h5'%(save_dir,dataset)
    _, _, xyz_err = get_err.get_err_from_normuvd(path=path,dataset=dataset,normuvd=normuvd[:,:,:],jnt_idx=palm_idx,setname=setname)

    return xyz_err


def train_all_dp(save_dir, version, in_dataset, test_set, 
                 partA, test_img0, test_target, k, pool_stage, 
                 MC_dropout, no_epochs, batch_size, lr, train_img0, train_target, method, palm_idx, MC, setname, entire_dataset, MC_VAR):
    
    joint_no = np.arange((21))
    aleatoric_monte_carlo_simulations = 100
    error_train = np.zeros((len(joint_no)))
    error_test = np.zeros((len(joint_no)))
    
    # Check if the set of weights already exists, otherwise train the deep prior network
    while True: #
    #while not os.path.exists('%s/deepprior/weight_mc_%s'%(save_dir,version+'_'+str(pool_stage)+'_'+str(k))):
        print(version)
        dims = 63
        keras.backend.clear_session()
        with tf.device('/gpu:'+ GPUv):
            if MC_VAR==0:
                model = hpe.deepprior_A(img_rows=128,img_cols=128,
                                        num_kern=12,num_f1=1024,cnn_out_dim=dims)
            if MC_VAR==1:
                model = hpe.deepprior_B(img_rows=128,img_cols=128,
                                        num_kern=12,num_f1=1024,cnn_out_dim=dims)                                               
            if MC_VAR==2:
                model = hpe.deepprior_C(img_rows=128,img_cols=128,
                                        num_kern=12,num_f1=1024,cnn_out_dim=dims)
            
            # model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
            model.compile(optimizer=Adam(lr=lr),
                    loss={"regress_variance": bayesian_loss_with_uncertainty(aleatoric_monte_carlo_simulations, dims),
                            "dense_3":mean_squared_errordp},metrics={'dense_3': metrics.mean_squared_error},
                            loss_weights={'regress_variance': 1., 'dense_3': 1.})

            model_json = model.to_json()
        # writer = tf.summary.FileWriter('logs')
        with open("%s/deepprior/%s.json"%(save_dir,version), "w") as json_file:
            json_file.write(model_json)
        
        if entire_dataset:
            train_img0, train_target,_,_ = load_data_dp(save_dir, in_dataset, joint_no)
        else:
            train_x0, train_y,_,_ = load_data_dp(save_dir, in_dataset+str(pool_stage), joint_no)
            
            if pool_stage == 1:
                if os.path.isfile("init_set_1k%s.npy"%setname):
                    partA = np.load("init_set_1k%s.npy"%setname)
                else:
                    data_idx = np.arange(int(train_x0.shape[0]))
                    np.random.shuffle(data_idx)
                    partA = data_idx[:ADDIM]
                    np.save("init_set_1k%s.npy"%setname,partA)
                train_img0 =  train_x0[partA]
                del train_x0
                train_target =  train_y[partA]
                del train_y
            else:
                if method == "random":
                    data_idx = np.arange(int(train_x0.shape[0]))
                    np.random.shuffle(data_idx)
                    partA = data_idx[:ADDIM]  
                if method == "unc":  
                    _, uncertainties = predict_features_dp(save_dir, version+'_'+str(pool_stage-1)+'_'+str(k), in_dataset, 
                                                                  joint_no, train_img0, pool_stage, lr, MC, MC_dropout, setname)     
                    uncertainties = uncertainties[:train_x0.shape[0],:]
                    # av_idx = np.arange(pool_size,pool_size+(pool_stage-1)*ADDIM)     
                    # unc_entropies = entropy(np.transpose(uncertainties), base=2)
                    unc_sum = np.sum(uncertainties, axis=-1)
                    # partA = np.argsort(np.amax(unc_mean, axis=0)* unc_entropies + unc_mean)[-ADDIM:]  
                    partA = np.argsort(unc_sum)[-ADDIM:]        
                if method == "kcg":
                    partA = geom_kcg(save_dir, in_dataset, version+'_'+str(pool_stage-1)+'_'+str(k), 
                                    train_x0.shape[0], pool_stage, ADDIM, train_img0, lr, joint_no, MC, MC_dropout, setname)
                if method == "cke":
                    partA = geom_cke(save_dir, in_dataset, version+'_'+str(pool_stage-1)+'_'+str(k), 
                                    train_x0.shape[0], pool_stage, ADDIM, train_img0, lr, joint_no, MC, MC_dropout, setname)
                train_img0 =  np.concatenate((train_img0, train_x0[partA]), axis=0)
                del train_x0
                train_target =  np.concatenate((train_target, train_y[partA]), axis=0)
                del train_y

        batch_sizep = batch_size#int(pool_stage*batch_size/10)
        n_train_batches=int(train_img0.shape[0]/batch_sizep)
        validfreq = int(n_train_batches / 3)
        # Baseline model

        # Save initial random weights to re-initialise at the end
        model.save_weights('initial_weights_dp_%s.h5'%setname)
        model_filepath='%s/deepprior/weight_mc_%s'%(save_dir,version+'_'+str(pool_stage)+'_'+str(k))

        best_lost = 999
        train_idx=range(train_img0.shape[0])
        epoch = 0
        done_looping=False

        # train_idx=range(train_idx.shape[0])
        test_cost=[best_lost]
        train_cost=[best_lost]
        # validfreq = 20
        num_iter=0
        val_loss = 999
        while (epoch < no_epochs) and (not done_looping):
            epoch +=1
            print('training @ epoch = ', epoch)
            train_idx=shuffle(train_idx)
            for minibatch_index in range(n_train_batches):
                num_iter+=1
                batch_idx = train_idx[minibatch_index * batch_sizep: (minibatch_index + 1) * batch_sizep]
                # x0,x1,x2,y=data_augmentation.augment_data_3d_mega_rot_scale(train_img0[batch_idx],train_img1[batch_idx],train_img2[batch_idx],
                #                                                     train_target[batch_idx])

                with tf.device('/gpu:'+ GPUv):
                    out = model.train_on_batch(x={'input0':train_img0[batch_idx]},y=[train_target[batch_idx],train_target[batch_idx]])

                if np.isnan(out).any():
                    continue
                    # exit('nan,%d,%d'%(epoch,minibatch_index))

                if (num_iter+1)%int(validfreq)==0:
                    model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                    # np.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                    with tf.device('/gpu:'+ GPUv):
                        val_loss = model.evaluate(x={'input0':test_img0},y=[test_target,test_target], batch_size=batch_size)

                    if np.isinf(val_loss).any():
                        model.save_weights('%s_inf'%model_filepath, overwrite=True)
                        # np.save('%s_inf'%history_filepath,[train_cost,test_cost])
                    # print('\n')
                    # print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                    test_cost.append(val_loss)
                    train_cost.append(out)
                    if val_loss[-1]<best_lost:
                        print('-'*30,model_filepath,'best val_loss',val_loss)
                        best_lost=val_loss[-1]
                        # Don't save weights and biases
                        model.save_weights(model_filepath, overwrite=True)
                        # np.save(history_filepath,[train_cost,test_cost])
        
        #del train_img0, train_target
        model.load_weights('initial_weights_dp_%s.h5'%setname)
        break

    if entire_dataset:
        error_train[joint_no] = predict_with_aleatoric_dp(save_dir,
                                                version+'_'+str(pool_stage)+'_'+str(k),
                                                in_dataset, joint_no, lr, MC_dropout, setname)
    else:
        error_train[joint_no] = predict_with_aleatoric_dp(save_dir,
                                                        version+'_'+str(pool_stage)+'_'+str(k),
                                                        in_dataset+str(pool_stage), joint_no, lr, MC_dropout, setname)
    error_test[joint_no] = predict_with_aleatoric_dp(save_dir,
                                                    version+'_'+str(pool_stage)+'_'+str(k),
                                                    test_set, joint_no, lr, MC_dropout, setname)
    return error_train, error_test,version+'_'+str(pool_stage)+'_'+str(k), train_img0, train_target


def train_dp_orig(save_dir, version, in_dataset, test_set, 
                 partA, test_img0, test_target, k, pool_stage, 
                 MC_dropout, no_epochs, batch_size, lr, train_img0, train_target, method, palm_idx, MC, setname, entire_dataset):
    
    joint_no = np.arange((21))
    # aleatoric_monte_carlo_simulations = 100
    error_train = np.zeros((len(joint_no)))
    error_test = np.zeros((len(joint_no)))
    
    # Check if the set of weights already exists, otherwise train the deep prior network
    while True: 
    #while not os.path.exists('%s/deepprior/weight_%s'%(save_dir,version+'_'+str(pool_stage)+'_'+str(k))):
        print(version)
        dims = 63
        keras.backend.clear_session()
        with tf.device('/gpu:'+ GPUv):
            model = hpe.deepprior(img_rows=128,img_cols=128,
                                  num_kern=12,num_f1=1024,cnn_out_dim=dims)
            
            # model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
            model.compile(optimizer=Adam(lr=lr),
                    loss={"dense_3":mean_squared_errordp},metrics={'dense_3': metrics.mean_squared_error})

            model_json = model.to_json()

        with open("%s/deepprior/%s.json"%(save_dir,version), "w") as json_file:
            json_file.write(model_json)
        
        #for d in in_dataset:
            # partA = unc_entropy_max1(save_dir, d, version+'_'+str(pool_stage)+'_'+str(k), partA)
        if entire_dataset:
            train_img0, train_target,_,_ = load_data_dp(save_dir, in_dataset, joint_no)
        else:
            train_x0, train_y,_,_ = load_data_dp(save_dir, in_dataset+str(pool_stage), joint_no)

            if pool_stage == 1:
                if os.path.isfile("init_set_1k%s.npy"%setname):
                    partA = np.load("init_set_1k%s.npy"%setname)
                else:
                    data_idx = np.arange(int(train_x0.shape[0]))
                    np.random.shuffle(data_idx)
                    partA = data_idx[:ADDIM]
                    np.save("init_set_1k%s.npy"%setname,partA)
                train_img0 =  train_x0[partA]
                del train_x0
                train_target =  train_y[partA]
                del train_y
            else:
                if method == "random":
                    data_idx = np.arange(int(train_x0.shape[0]))
                    np.random.shuffle(data_idx)
                    partA = data_idx[:ADDIM]
                    # partA = np.argsort(np.amax(unc_mean, axis=0)* unc_entropies + unc_mean)[-ADDIM:]
                if method == "kcg":
                    partA = geom_kcg(save_dir, in_dataset, version+'_'+str(pool_stage-1)+'_'+str(k), 
                                    train_x0.shape[0], pool_stage, ADDIM, train_img0, lr, joint_no, MC, MC_dropout, setname)

                train_img0 =  np.concatenate((train_img0, train_x0[partA]), axis=0)
                del train_x0
                train_target =  np.concatenate((train_target, train_y[partA]), axis=0)
                del train_y

        batch_sizep = batch_size#int(pool_stage*batch_size/10)
        n_train_batches=int(train_img0.shape[0]/batch_sizep)
        validfreq = int(n_train_batches / 2)
        # Baseline model

        # Save initial random weights to re-initialise at the end
        model.save_weights('initial_weights_dpo_%s.h5'%setname)
        model_filepath='%s/deepprior/weight_%s'%(save_dir,version+'_'+str(pool_stage)+'_'+str(k))

        best_lost = 999
        train_idx=range(train_img0.shape[0])
        epoch = 0
        done_looping=False

        # train_idx=range(train_idx.shape[0])
        test_cost=[best_lost]
        train_cost=[best_lost]
        # validfreq = 20
        num_iter=0
        val_loss = 999
        while (epoch < no_epochs) and (not done_looping):
            epoch +=1
            print('training @ epoch = ', epoch)
            train_idx=shuffle(train_idx)
            for minibatch_index in range(n_train_batches):
                num_iter+=1
                batch_idx = train_idx[minibatch_index * batch_sizep: (minibatch_index + 1) * batch_sizep]
                # x0,x1,x2,y=data_augmentation.augment_data_3d_mega_rot_scale(train_img0[batch_idx],train_img1[batch_idx],train_img2[batch_idx],
                #                                                     train_target[batch_idx])

                with tf.device('/gpu:'+ GPUv):
                    out = model.train_on_batch(x={'input0':train_img0[batch_idx]},y=train_target[batch_idx])

                if np.isnan(out).any():
                    continue
                    # exit('nan,%d,%d'%(epoch,minibatch_index))

                if (num_iter+1)%int(validfreq)==0:
                    model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                    # np.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                    with tf.device('/gpu:'+ GPUv):
                        val_loss = model.evaluate(x={'input0':test_img0},y=test_target, batch_size=batch_size)

                    if np.isinf(val_loss).any():
                        model.save_weights('%s_inf'%model_filepath, overwrite=True)
                        # np.save('%s_inf'%history_filepath,[train_cost,test_cost])
                    # print('\n')
                    # print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                    test_cost.append(val_loss)
                    train_cost.append(out)
                    if val_loss[-1]<best_lost:
                        print('-'*30,model_filepath,'best val_loss',val_loss)
                        best_lost=val_loss[-1]
                        # Don't save weights and biases
                        model.save_weights(model_filepath, overwrite=True)
                        # np.save(history_filepath,[train_cost,test_cost])
        
        #del train_img0, train_target
        model.load_weights('initial_weights_dpo_%s.h5'%setname)
        break

    if entire_dataset:
        error_train[joint_no] = predict_dp(save_dir,
                                        version+'_'+str(pool_stage)+'_'+str(k),
                                        in_dataset, joint_no, lr, MC_dropout, setname)
    else:
        error_train[joint_no] = predict_dp(save_dir,
                                        version+'_'+str(pool_stage)+'_'+str(k),
                                        in_dataset+str(pool_stage), joint_no, lr, MC_dropout, setname)
    error_test[joint_no] = predict_dp(save_dir,
                                    version+'_'+str(pool_stage)+'_'+str(k),
                                    test_set, joint_no, lr, MC_dropout, setname)
    return error_train, error_test,version+'_'+str(pool_stage)+'_'+str(k), train_img0, train_target