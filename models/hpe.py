from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D,BatchNormalization,Dense,Flatten,Dropout, Reshape
from keras.layers import * 
from keras import applications
from keras.initializers import glorot_uniform

def deepprior(img_rows,img_cols,num_kern,num_f1,cnn_out_dim):
    padding='valid'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    conv1 = Conv2D(12, (5,5), activation=None, padding=padding)(inputs_r0) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    activ1 = LeakyReLU(alpha=0.05)(pool1)

    conv2 = Conv2D(12, (5,5), activation=None, padding=padding)(activ1) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    activ2 = LeakyReLU(alpha=0.05)(pool2)

    conv3 = Conv2D(12, (5,5), activation=None, padding=padding)(activ2) 
    activ3 = LeakyReLU(alpha=0.05)(conv3)

    convout = Flatten()(activ3)
    activation  ='relu'
    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet11 = Dense(num_f1,activation=activation)(fullconnet1)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)

    model = Model(inputs=[inputs_r0], outputs=reg_out)
    # print(model.summary())
    return model


def deepprior_A(img_rows,img_cols,num_kern,num_f1,cnn_out_dim):
    padding='valid'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    conv1 = Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(inputs_r0), training=True) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    activ1 = LeakyReLU(alpha=0.05)(pool1)

    conv2 =  Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(activ1), training=True) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    activ2 = LeakyReLU(alpha=0.05)(pool2)

    conv3 =  Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(activ2), training=True) 
    activ3 = LeakyReLU(alpha=0.05)(conv3)

    convout = Flatten()(activ3)
    activation  ='relu'
    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    # fullconnet1 = Dropout(0.3)(fullconnet1, training=True)
    fullconnet11 = Dense(num_f1,activation=activation)(fullconnet1)
    # fullconnet11 = Dropout(0.3)(fullconnet11, training=True)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)
    aleatoric_variance =  Dense(cnn_out_dim,activation=None)(fullconnet11)

    output = concatenate([reg_out,aleatoric_variance], name='regress_variance')
    model = Model(inputs=[inputs_r0], outputs=[output,reg_out])
    # print(model.summary())
    return model

def deepprior_B(img_rows,img_cols,num_kern,num_f1,cnn_out_dim):
    padding='valid'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    # conv1 = Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(inputs_r0), training=True) 
    conv1 = Conv2D(12, (5,5), activation=None, padding=padding)(inputs_r0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    activ1 = LeakyReLU(alpha=0.05)(pool1)

    conv2 =  Conv2D(12, (5,5), activation=None, padding=padding)(activ1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    activ2 = LeakyReLU(alpha=0.05)(pool2)

    conv3 =  Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(activ2), training=True) 
    activ3 = LeakyReLU(alpha=0.05)(conv3)

    convout = Flatten()(activ3)
    activation  ='relu'
    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet1 = Dropout(0.3)(fullconnet1, training=True)
    fullconnet11 = Dense(num_f1,activation=activation)(fullconnet1)

    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)
    aleatoric_variance =  Dense(cnn_out_dim,activation=None)(fullconnet11)

    output = concatenate([reg_out,aleatoric_variance], name='regress_variance')
    model = Model(inputs=[inputs_r0], outputs=[output,reg_out])
    # print(model.summary())
    return model

def deepprior_C(img_rows,img_cols,num_kern,num_f1,cnn_out_dim):
    padding='valid'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    conv1 = Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(inputs_r0), training=True) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    activ1 = LeakyReLU(alpha=0.05)(pool1)

    conv2 =  Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(activ1), training=True) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    activ2 = LeakyReLU(alpha=0.05)(pool2)

    conv3 =  Dropout(0.3)(Conv2D(12, (5,5), activation=None, padding=padding)(activ2), training=True) 
    activ3 = LeakyReLU(alpha=0.05)(conv3)

    convout = Flatten()(activ3)
    activation  ='relu'
    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet1 = Dropout(0.3)(fullconnet1, training=True)
    fullconnet11 = Dense(num_f1,activation=activation)(fullconnet1)
    fullconnet11 = Dropout(0.3)(fullconnet11, training=True)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)
    aleatoric_variance =  Dense(cnn_out_dim,activation=None)(fullconnet11)

    output = concatenate([reg_out,aleatoric_variance], name='regress_variance')
    model = Model(inputs=[inputs_r0], outputs=[output,reg_out])
    # print(model.summary())
    return model

