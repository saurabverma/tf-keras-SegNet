from keras.models import Model
from keras.layers import Input, MaxPooling2D, Dropout, concatenate, Convolution2D, Activation, Reshape, BatchNormalization, UpSampling2D, Concatenate, AveragePooling2D, Flatten, Dense
# from keras.layers.core import Activation, Reshape
# from keras.layers.convolutional import 
from keras.utils.training_utils import multi_gpu_model
# from keras.layers.normalization import BatchNormalization

from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D




def segnet(
	input_shape,
	n_labels,
	kernel=3,
	pool=2,
	output_mode="softmax",
	gpus=1):
	# encoder
	inputs = Input(shape=input_shape)

	conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation("relu")(conv_1)
	conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Activation("relu")(conv_2)

	pool_1, mask_1 = MaxPoolingWithArgmax2D((pool, pool))(conv_2)

	conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation("relu")(conv_3)
	conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Activation("relu")(conv_4)

	pool_2, mask_2 = MaxPoolingWithArgmax2D((pool, pool))(conv_4)

	conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation("relu")(conv_5)
	conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation("relu")(conv_6)
	conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Activation("relu")(conv_7)

	pool_3, mask_3 = MaxPoolingWithArgmax2D((pool, pool))(conv_7)

	conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation("relu")(conv_8)
	conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation("relu")(conv_9)
	conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Activation("relu")(conv_10)

	pool_4, mask_4 = MaxPoolingWithArgmax2D((pool, pool))(conv_10)

	conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation("relu")(conv_11)
	conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation("relu")(conv_12)
	conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation("relu")(conv_13)

	pool_5, mask_5 = MaxPoolingWithArgmax2D((pool, pool))(conv_13)
	print("Build encoder done..")

	# decoder

	unpool_1 = MaxUnpooling2D((pool, pool))([pool_5, mask_5])

	conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation("relu")(conv_14)
	conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation("relu")(conv_15)
	conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation("relu")(conv_16)

	unpool_2 = MaxUnpooling2D((pool, pool))([conv_16, mask_4])

	conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation("relu")(conv_17)
	conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation("relu")(conv_18)
	conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation("relu")(conv_19)

	unpool_3 = MaxUnpooling2D((pool, pool))([conv_19, mask_3])

	conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation("relu")(conv_20)
	conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation("relu")(conv_21)
	conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation("relu")(conv_22)

	unpool_4 = MaxUnpooling2D((pool, pool))([conv_22, mask_2])

	conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation("relu")(conv_23)
	conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation("relu")(conv_24)

	unpool_5 = MaxUnpooling2D((pool, pool))([conv_24, mask_1])

	conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation("relu")(conv_25)

	conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
	conv_26 = BatchNormalization()(conv_26)
	conv_26 = Reshape((input_shape[0]*input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

	outputs = Activation(output_mode)(conv_26)
	print("Build decoder done..")

	model = Model(inputs=inputs, outputs=outputs, name="SegNet")

	# make the model parallel
	if gpus>1:
		model = multi_gpu_model(model, gpus=gpus)

	return model










def unet(input_shape,
	n_labels,
	kernel=3,
	pool=2,
	output_mode='sigmoid',
	gpus=1):
	# encoder
	inputs = Input(shape=input_shape)

	conv1 = Convolution2D(64, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Convolution2D(64, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(pool, pool))(conv1)

	conv2 = Convolution2D(128, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Convolution2D(128, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(pool, pool))(conv2)

	conv3 = Convolution2D(256, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Convolution2D(256, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(pool, pool))(conv3)

	conv4 = Convolution2D(512, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Convolution2D(512, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(pool, pool))(drop4)

	conv5 = Convolution2D(1024, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Convolution2D(1024, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)
	print("Build encoder done..")

	# decoder


	up6 = Convolution2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(pool, pool))(drop5))
	merge6 = concatenate([drop4, up6], axis=3)
	conv6 = Convolution2D(512, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Convolution2D(512, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Convolution2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(pool, pool))(conv6))
	merge7 = concatenate([conv3, up7], axis=3)
	conv7 = Convolution2D(256, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Convolution2D(256, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Convolution2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(pool, pool))(conv7))
	merge8 = concatenate([conv2, up8], axis=3)
	conv8 = Convolution2D(128, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Convolution2D(128, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Convolution2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(pool, pool))(conv8))
	merge9 = concatenate([conv1, up9], axis=3)
	conv9 = Convolution2D(64, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Convolution2D(64, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Convolution2D(2, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

	conv10 = Convolution2D(1, 1, activation=output_mode)(conv9)
	print("Build decoder done..")

	model = Model(input=inputs, output=conv10)

	# make the model parallel
	if gpus > 1:
		model = multi_gpu_model(model, gpus=gpus)

	return model










def segunet(
	input_shape,
	n_labels,
	kernel=3,
	pool=2,
	output_mode="softmax",
	gpus=1):

	inputs = Input(shape=input_shape)

	# Encoder
	conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation("relu")(conv_1)
	conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Activation("relu")(conv_2)

	pool_1, mask_1 = MaxPoolingWithArgmax2D((pool, pool))(conv_2)

	conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation("relu")(conv_3)
	conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Activation("relu")(conv_4)

	pool_2, mask_2 = MaxPoolingWithArgmax2D((pool, pool))(conv_4)

	conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation("relu")(conv_5)
	conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation("relu")(conv_6)
	conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Activation("relu")(conv_7)

	pool_3, mask_3 = MaxPoolingWithArgmax2D((pool, pool))(conv_7)

	conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation("relu")(conv_8)
	conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation("relu")(conv_9)
	conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Activation("relu")(conv_10)

	pool_4, mask_4 = MaxPoolingWithArgmax2D((pool, pool))(conv_10)

	conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation("relu")(conv_11)
	conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation("relu")(conv_12)
	conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation("relu")(conv_13)

	pool_5, mask_5 = MaxPoolingWithArgmax2D((pool, pool))(conv_13)
	print("Build enceder done..")

	# between encoder and decoder
	conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(pool_5)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation("relu")(conv_14)
	conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation("relu")(conv_15)
	conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation("relu")(conv_16)

	# decoder
	unpool_1 = MaxUnpooling2D((pool, pool))([conv_16, mask_5])
	concat_1 = Concatenate()([unpool_1, conv_13])

	conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(concat_1)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation("relu")(conv_17)
	conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation("relu")(conv_18)
	conv_19 = Convolution2D(512, (kernel, kernel), padding="same")(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation("relu")(conv_19)

	unpool_2 = MaxUnpooling2D((pool, pool))([conv_19, mask_4])
	concat_2 = Concatenate()([unpool_2, conv_10])

	conv_20 = Convolution2D(512, (kernel, kernel), padding="same")(concat_2)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation("relu")(conv_20)
	conv_21 = Convolution2D(512, (kernel, kernel), padding="same")(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation("relu")(conv_21)
	conv_22 = Convolution2D(256, (kernel, kernel), padding="same")(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation("relu")(conv_22)

	unpool_3 = MaxUnpooling2D((pool, pool))([conv_22, mask_3])
	concat_3 = Concatenate()([unpool_3, conv_7])

	conv_23 = Convolution2D(256, (kernel, kernel), padding="same")(concat_3)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation("relu")(conv_23)
	conv_24 = Convolution2D(256, (kernel, kernel), padding="same")(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation("relu")(conv_24)
	conv_25 = Convolution2D(128, (kernel, kernel), padding="same")(conv_24)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation("relu")(conv_25)

	unpool_4 = MaxUnpooling2D((pool, pool))([conv_25, mask_2])
	concat_4 = Concatenate()([unpool_4, conv_4])

	conv_26 = Convolution2D(128, (kernel, kernel), padding="same")(concat_4)
	conv_26 = BatchNormalization()(conv_26)
	conv_26 = Activation("relu")(conv_26)
	conv_27 = Convolution2D(64, (kernel, kernel), padding="same")(conv_26)
	conv_27 = BatchNormalization()(conv_27)
	conv_27 = Activation("relu")(conv_27)

	unpool_5 = MaxUnpooling2D((pool, pool))([conv_27, mask_1])
	concat_5 = Concatenate()([unpool_5, conv_2])

	conv_28 = Convolution2D(64, (kernel, kernel), padding="same")(concat_5)
	conv_28 = BatchNormalization()(conv_28)
	conv_28 = Activation("relu")(conv_28)

	conv_29 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_28)
	conv_29 = BatchNormalization()(conv_29)
	conv_29 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_29)

	outputs = Activation(output_mode)(conv_29)
	print("Build decoder done..")

	model = Model(inputs=inputs, outputs=outputs, name="SegUNet")

	# make the model parallel
	if gpus > 1:
		model = multi_gpu_model(model, gpus=gpus)

	return model










# A single Unit - FastNet
def UnitCell(x, channels, kernel_size=[3, 3], strides=(1, 1)):
	y = BatchNormalization(scale=True, momentum=0.95)(x)
	y = Activation("relu")(y)
	y = Convolution2D(channels, kernel_initializer='he_normal',
	                  kernel_size=kernel_size, strides=(strides), padding="same")(y)

	return y

# The whole network - FastNet

def fastnet(
	input_shape,
	n_labels,
	kernel = 3,
	pool = 2,
	output_mode = "softmax",
	gpus = 1):

	inputs = Input(input_shape)

	y = UnitCell(inputs, 64, kernel_size=[kernel, kernel])
	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = MaxPooling2D(pool_size=(pool, pool), strides=[2, 2])(y)

	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = UnitCell(y, 128, kernel_size=[kernel, kernel])

	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = MaxPooling2D(pool_size=(pool, pool), strides=[2, 2])(y)

	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = UnitCell(y, 128, kernel_size=[kernel, kernel])

	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = MaxPooling2D(pool_size=(pool, pool), strides=[2, 2])(y)

	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = UnitCell(y, 128, kernel_size=[kernel, kernel])
	y = MaxPooling2D(pool_size=(pool, pool), strides=[2, 2])(y)

	y = UnitCell(y, 128, kernel_size=[1, 1]) # kernel maintained low intensionally
	y = UnitCell(y, 128, kernel_size=[1, 1]) # kernel maintained low intensionally
	y = UnitCell(y, 128, kernel_size=[1, 1]) # kernel maintained low intensionally
	y = AveragePooling2D(pool_size=(pool, pool))(y)
	# y = Flatten()(y)
	# outputs = Dense(n_labels, activation=output_mode)(y)
	# y = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(y)
	# outputs = Activation(output_mode)(y)
	print("Build FastNet done..")


	model = Model(inputs=inputs, outputs=outputs, name="FastNet")

	# make the model parallel
	if gpus > 1:
		model = multi_gpu_model(model, gpus=gpus)

	return model
