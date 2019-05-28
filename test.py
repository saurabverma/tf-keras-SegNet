#!usr/bin/env python3

import argparse
import pandas as pd
import yaml
from keras.callbacks import TensorBoard
from keras.optimizers import Adadelta
from time import time

from  model import unet, segnet, segunet
from generator import data_gen_small


def argparser():
	# command line argments
	parser = argparse.ArgumentParser(description="SegNet LIP dataset")
 
	parser.add_argument("--model", required=True, choices=['unet', 'segnet', 'segunet'], help="model to use")
	parser.add_argument("--test_list", required=True, help="file containing a list of test images (jpg)")
	parser.add_argument("--testimg_dir", required=True, help="directory containing test images")
	parser.add_argument("--testmsk_dir", required=True, help="directory containing groundtruth of test images")
	parser.add_argument("--model_weights", required=True, help="file containing model weights to the examined")
	parser.add_argument("--save_dir", required=True, help="folder where tensorboard output is to be placed")

	parser.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="# of GPUs used for training")
	parser.add_argument("--gpus", type=int, default=1, help="# of GPUs used for training")
	parser.add_argument("--batch_size", default=10, type=int, help="batch size")
	parser.add_argument("--epoch_steps", default=100, type=int, help="number of epoch step")
	parser.add_argument("--n_labels", default=20,type=int, help="Number of label")
	parser.add_argument("--input_shape", default=(256, 256, 3), help="Input images shape")
	parser.add_argument("--kernel", default=3, type=int, help="Kernel size")
	parser.add_argument("--pool_size", default=(2, 2), type=tuple, help="pooling and unpooling size")
	parser.add_argument("--output_mode", default="softmax", type=str, help="output activation")

	loss_options = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
	parser.add_argument("--loss", default="categorical_crossentropy", choices=loss_options, help="loss function")
	# # TODO: optimizer options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam
	# parser.add_argument("--optimizer", default="adadelta", type=str, help="oprimizer")
	metric_options = ['binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy', 'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
	parser.add_argument("--metrics", default='categorical_accuracy', choices=metric_options, help="metrics")

	args = parser.parse_args()

	return args



def main(args):
	# set the necessary list i.e. list of names of images to be read
	test_list = pd.read_csv(args.test_list, header=None)

	# set the necessary directories
	testimg_dir = args.testimg_dir
	testmsk_dir = args.testmsk_dir

	# Generate batch data for SGD
	# NOTE: This helps control the batch size for each test set
	test_gen = data_gen_small(testimg_dir, testmsk_dir, test_list, args.batch_size, [args.input_shape[0], args.input_shape[1]], args.n_labels)

	# Create a model
	if args.model == 'unet':
		model = unet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool_size, args.output_mode, args.gpus)
	elif args.model == 'segnet':
		model = segnet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool_size, args.output_mode, args.gpus)
	elif args.model == 'segunet':
		model = segunet(\
			args.input_shape, args.n_labels, args.kernel, \
			args.pool_size, args.output_mode, args.gpus)

	# TODO: Configure the model for training
	optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.01) # NOTE: The details doesn't matter here because we are not training the model
	model.compile(loss=args.loss, optimizer=optimizer, metrics=[args.metrics])

	# Set model trained weights
	model.load_weights(args.model_weights)

	# Keras moodel summary, for confirmation
	print(model.summary())

	# Test the model on data generated batch-by-batch by a Python generator
	# NOTE: We use evaluate_generator because we do provide our generated dataset with specific batch size
	tensorboard = TensorBoard(log_dir=args.save_dir)
	fit_start_time = time()
	model.evaluate_generator(test_gen, steps=args.epoch_steps, verbose=args.verbose, callbacks=[tensorboard])
	print('### Model fit time (s): ', time()-fit_start_time)



if __name__ == "__main__":
	program_start_time = time()
	args = argparser()
	main(args)
	print('### Whole program time (s): ', time()-program_start_time)
