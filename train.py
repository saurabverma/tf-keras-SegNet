#!usr/bin/env python3

import argparse
import pandas as pd
import yaml
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adadelta
from keras.utils.training_utils import multi_gpu_model
from time import time

from  model import unet, segnet, segunet, fastnet
from generator import data_gen_small


def argparser():
	# command line argments
	parser = argparse.ArgumentParser(description="Image segmentation")
 
	parser.add_argument("--model", required=True, choices=['unet', 'segnet', 'segunet', 'fastnet'], help="model to use")
	parser.add_argument("--save_dir", required=True, help="output directory")
	parser.add_argument("--train_list", required=True, help="file containing a list of train images (jpg)")
	parser.add_argument("--trainimg_dir", required=True, help="directory containing train images")
	parser.add_argument("--trainmsk_dir", required=True, help="directory containing groundtruth of train images")
	parser.add_argument("--val_list", required=True, help="file containing a list of val images (jpg)")
	parser.add_argument("--valimg_dir", required=True, help="directory containing train images")
	parser.add_argument("--valmsk_dir", required=True, help="directory containing groundtruth of val images")

	parser.add_argument("--initial_weights", type=str, default='', help="file containing initial weights to train upon")
	parser.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="# of GPUs used for training")
	parser.add_argument("--gpus", type=int, default=1, help="# of GPUs used for training")
	parser.add_argument("--period", type=int, default=1, help="epoch period for checkpoint")
	# NOTE: batch_size is limited by memory available; start with a high number and keep decreasing until the program can run without memory issues
	# NOTE: batch_size * epoch_steps = # training samples
	parser.add_argument("--batch_size", default=10, type=int, help="batch size")
	# parser.add_argument("--epoch_steps", default=100, type=int, help="number of epoch step")
	# NOTE: 1 epoch = 1 forward pass + 1 backward pass on ALL training samples
	parser.add_argument("--n_epochs", default=10, type=int, help="number of epoch")
	# NOTE: If we have time to go through all the validation samples, keep 'val_steps' to number of val samples (though this requires a lot of time and the system looks to be stuck after every epoch)
	# NOTE: batch_size * val_steps = # validation samples
	# parser.add_argument("--val_steps", default=10, type=int, help="number of valdation step")
	parser.add_argument("--n_labels", default=20,type=int, help="Number of label")
	parser.add_argument("--input_shape", default=(256, 256, 3), help="Input images shape")
	parser.add_argument("--kernel", default=3, type=int, help="Kernel size")
	parser.add_argument("--pool", default=2, type=int, help="pooling and unpooling size")
	parser.add_argument("--output_mode", default="softmax", type=str, help="output activation")

	parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
	parser.add_argument("--rho", default=0.95, type=float, help="Adadelta decay factor")
	parser.add_argument("--decay", default=0.01, type=float, help="Decay rate of Initial learning rate")
	parser.add_argument("--epsilon", default=None, type=float, help="Adadelta fuzz factor. If None, defaults to K.epsilon()")

	loss_options = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
	parser.add_argument("--loss", default="categorical_crossentropy", choices=loss_options, help="loss function")
	# # TODO: optimizer options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam
	# parser.add_argument("--optimizer", default="adadelta", type=str, help="oprimizer")
	metric_options = ['accuracy', 'binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy', 'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
	parser.add_argument("--metrics", default='accuracy', choices=metric_options, help="metrics")

	args = parser.parse_args()

	return args



def main(args):
	# set the necessary list i.e. list of names of images to be read
	train_list = pd.read_csv(args.train_list, header=None)
	train_list_len = len(train_list)
	epoch_steps = int(train_list_len/args.batch_size)
	val_list = pd.read_csv(args.val_list, header=None)
	val_list_len = len(val_list)
	val_steps = int(val_list_len/args.batch_size)

	# set the necessary directories
	trainimg_dir = args.trainimg_dir
	trainmsk_dir = args.trainmsk_dir
	valimg_dir = args.valimg_dir
	valmsk_dir = args.valmsk_dir

	# Generate batch data for SGD
	# NOTE: This helps control the batch size for each training and validation set
	train_gen = data_gen_small(trainimg_dir, trainmsk_dir, train_list, args.batch_size, [args.input_shape[0], args.input_shape[1]], args.n_labels)
	val_gen = data_gen_small(valimg_dir, valmsk_dir, val_list, args.batch_size, [args.input_shape[0], args.input_shape[1]], args.n_labels)

	# Create a model
	if args.model == 'unet':
		model = unet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool, args.output_mode, args.gpus)
	elif args.model == 'segnet':
		model = segnet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool, args.output_mode, args.gpus)
	elif args.model == 'segunet':
		model = segunet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool, args.output_mode, args.gpus)
	elif args.model == 'fastnet':
		model = fastnet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool, args.output_mode, args.gpus)

	# Keras moodel summary
	print(model.summary())

	# TODO: Configure the model for training
	optimizer = Adadelta(lr=args.lr, rho=args.rho, epsilon=args.epsilon, decay=args.decay)
	model.compile(loss=args.loss, optimizer=optimizer, metrics=[args.metrics])
	# model.compile(loss=args.loss, optimizer=args.optimizer, metrics=[args.metrics])

	# If pre-trained weights available, use those
	if(args.initial_weights):
		model.load_weights(args.initial_weights)
		print('Initial weights loaded')

	# Generate log for tensorboard (currently only useful parameters are added to the dir name)
	log_dir = args.save_dir + \
		'model_' + args.model + \
		'_batch_size_' + str(args.batch_size) + \
		'_epoch_steps_' + str(epoch_steps) + \
		'_n_epochs_' + str(args.n_epochs) + \
		'_lr_' + str(args.lr) + \
    	'_decay_' + str(args.decay) + \
    	'_labels_' + str(args.n_labels) + \
    	'/'
	tensorboard = TensorBoard(log_dir=log_dir)

	# Generate checkpoints
	checkpoint = ModelCheckpoint(log_dir+'weights_at_epoch_{epoch:03d}.hdf5', verbose=1, save_best_only=False, save_weights_only=True, period=args.period) # Create 10 checkpoints only

	# Train the model on data generated batch-by-batch by a Python generator
	# NOTE: We use fit_generator because we do provide our generated dataset with specific batch size
	fit_start_time = time()
	model.fit_generator(train_gen, steps_per_epoch=epoch_steps, epochs=args.n_epochs, validation_data=val_gen, validation_steps=val_steps, verbose=args.verbose, callbacks=[checkpoint, tensorboard])
	print('### Model fit time (s): ', time()-fit_start_time)

	# # NOTE: Cannot save the whole model OR the model structure if the model is not Keras Sequential, it throws "AttributeError: 'NoneType' object has no attribute 'update'"
	# # This is a bug in Keras (not sure about TensorFlow)
	# # Therefore, the model strutuce must be generated every time it is required to be implemented outside of the current script
	# model.save(log_dir + 'model.hdf5') # Model architecture and final weights

	# Save final weights
	model.save_weights(log_dir + 'weights_at_epoch_%03d.hdf5' % (args.n_epochs))
	print('Final model weights saved.')



if __name__ == "__main__":
	program_start_time = time()
	args = argparser()
	main(args)
	print('### Whole program time (s): ', time()-program_start_time)
