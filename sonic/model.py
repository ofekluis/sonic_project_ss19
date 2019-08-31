from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras import backend as K


def ddqn_model(input_shape, nb_classes, info = None, include_top=True, weights=None):
    """info is an extra feature vector to combine with the picture's pixels"""
    x_in = Input(shape=input_shape, name='x_input', dtype='float32')
    info_input = Input(shape=(info,), name='info_input', dtype='float32')

    x = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], input_shape=input_shape)(x_in)
    x = Conv2D(filters=64, kernel_size=[4,4],strides=[2,2])(x)
    x = Conv2D(filters=64, kernel_size=[3,3],strides=[1,1])(x)
    x = Conv2D(filters=64, kernel_size=[7,7],strides=[1,1])(x)

    x = Flatten()(x)
    # add info to the conv output
    x = Concatenate(name='concatenation')([x, info_input])

    #predict a q value for each possible action in the state
    adv_dense =  Dense(512, activation="relu", kernel_initializer='glorot_uniform')(x)
    advantage = Dense(nb_classes, activation="relu", kernel_initializer='glorot_uniform')(adv_dense)

    #predict one state value for the state
    v_dense = Dense(512, activation="relu", kernel_initializer='glorot_uniform')(x)
    value = Dense(1, activation="relu", kernel_initializer='glorot_uniform')(v_dense)

    # concatenate state value and advantage by adding the normalized action value to
    # the state value for each action
    x =  Lambda(lambda x: x[0]- K.mean(x[0])+x[1])([advantage, value])

    model = Model(inputs=[x_in,info_input], outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model
