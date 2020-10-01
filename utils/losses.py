from keras import backend as K

def mean_squared_errordp(true, pred):
    return K.mean((pred[:,0:63] - true)**2)

def gaussian_squared_error(true, pred, dist, dims):
    def map_fn(i):
        # std_samples = K.transpose(dist.sample(num_joint_coord))
        std_samples = dist.sample()
        distorted_loss =K.square(pred[:,0:dims] + std_samples  - true) 
        return distorted_loss
    return map_fn


# Bayesian regression loss function
# N data points
# true - true values. Shape: (N)
# pred - predicted values (mean, log(variance)). Shape: (N, 2)
# returns - losses. Shape: (N)
def bayesian_loss_with_uncertainty(aleatoric_MC_simulations,dims):
    def loss_with_uncertainty(true, mean_pred):
        # Compute the exponential as the variance is in logarithmic scale
        # aleatoric_MC_simulations = 100
        aleatoric_variance = mean_pred[:,dims:]
        undistorted_loss = K.abs(mean_pred[:,0:dims] - true)
        final_loss = K.mean(.5 * undistorted_loss * K.exp(-aleatoric_variance) + .5 * aleatoric_variance, axis=1)  
        return final_loss
    return loss_with_uncertainty
