from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *


def get_post(x_in):
    x = LeakyReLU()(x_in)
    x = BatchNormalization()(x)
    return x

def get_block(x_in, ch_in, ch_out):
    x = Conv2D(ch_in,
               kernel_size=(1, 1),
               padding='same',
               use_bias=True)(x_in)
    x = get_post(x)

    x = DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=True)(x)
    x = get_post(x)
    x = MaxPool2D(pool_size=(2, 1),
                  strides=(2, 1))(x) # Separable pooling

    x = DepthwiseConv2D(kernel_size=(3, 1),
						padding='same',
						use_bias=True)(x)
    x = get_post(x)

    x = Conv2D(ch_out,
               kernel_size=(2, 1),
               strides=(1, 2),
               padding='same',
               use_bias=True)(x)
    x = get_post(x)

    return x


def Effnet(input_shape, nb_classes, info = None, include_top=True, weights=None):
	"""info is an extra feature vector to combine with the picture's pixels"""
	x_in = Input(shape=input_shape, name='x_input', dtype='float32')
	info_input = Input(shape=(info,), name='info_input', dtype='float32')
	#x = get_block(x_in, 16, 32)
	#x = get_block(x, 32, 64)
	#x = get_block(x, 64, 128)
	x = Conv2D(32, kernel_size=(8,8), strides = 4, activation="relu", input_shape=(128,128,4))(x_in)
	x = Conv2D(64, kernel_size=(4,4), strides = 2, activation="relu")(x)
	x = Conv2D(64, (3,3), activation="relu")(x)
	#x = Conv2D(32, (8, 4), activation='elu', input_shape=(128,128,4))(x_in)
	#x = Conv2D(64, (3, 2), activation='elu')(x)
	#x = Conv2D(64, (3, 2), activation='elu')(x)
	x = Flatten()(x)
	x = Concatenate(name='concatenation')([x, info_input])
	x = Dense(512, activation="relu", kernel_initializer='glorot_uniform')(x)
	x = Dense(nb_classes, kernel_initializer="uniform", activation="linear")(x)

	#x = Dense(nb_classes, activation='softmax')(x)
	model = Model(inputs=[x_in,info_input], outputs=x)

	if weights is not None:
		model.load_weights(weights, by_name=True)

	return model
