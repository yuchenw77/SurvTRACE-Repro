import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt

from pycox.datasets import metabric
from pycox.datasets import support
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

import pdb
from collections import defaultdict

from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceSingle
from survtrace.train_utils import Trainer
from survtrace.config import STConfig

def preprocess_support():
	df_train = support.read_df()
	df_test = df_train.sample(frac=0.3)
	df_train = df_train.drop(df_test.index)
	df_val = df_train.sample(frac=0.1)
	df_train = df_train.drop(df_val.index)

	cols_categorical = ["x1", "x2", "x3", "x4", "x5", "x6"]
	cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

	standardize = [([col], StandardScaler()) for col in cols_standardize]
	categorical = [([col], LabelEncoder()) for col in cols_categorical]
	x_mapper = DataFrameMapper(standardize + categorical)

	x_train = x_mapper.fit_transform(df_train).astype('float32')
	x_val = x_mapper.transform(df_val).astype('float32')
	x_test = x_mapper.transform(df_test).astype('float32')


def preprocess_metabric():
	df_train = metabric.read_df()
	df_test = df_train.sample(frac=0.3)
	df_train = df_train.drop(df_test.index)
	df_val = df_train.sample(frac=0.1)
	df_train = df_train.drop(df_val.index)

	# standardize numerical values.
	# leave binary values as-is
	cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
	cols_leave = ['x4', 'x5', 'x6', 'x7']

	standardize = [([col], StandardScaler()) for col in cols_standardize]
	leave = [(col, None) for col in cols_leave]

	x_mapper = DataFrameMapper(standardize + leave)

	x_train = x_mapper.fit_transform(df_train).astype('float32')
	x_val = x_mapper.transform(df_val).astype('float32')
	x_test = x_mapper.transform(df_test).astype('float32')

	# label transforms
	num_durations = 10
	labtrans = DeepHitSingle.label_transform(num_durations)
	get_target = lambda df: (df['duration'].values, df['event'].values)
	y_train = labtrans.fit_transform(*get_target(df_train))
	y_val = labtrans.transform(*get_target(df_val))

	# We don't need to transform the test labels
	durations_test, events_test = get_target(df_test)

	train = (x_train, y_train)
	val = (x_val, y_val)
	test = (x_test, durations_test, events_test)
	return train, val, test, labtrans

def train_deephit(train, val, labtrans):
	x_train, y_train = train
	in_features = x_train.shape[1]
	num_nodes = [32, 32]
	out_features = labtrans.out_features
	batch_norm = True
	dropout = 0.1

	net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
	model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
	batch_size = 256
	# set learning rate
	model.optimizer.set_lr(0.01)
	epochs = 100
	# early stop when validation loss stop improving
	callbacks = [tt.callbacks.EarlyStopping()]
	print(x_train.shape)
	print(len(y_train))
	print(type(y_train[0]))
	print(len(y_train[0]))
	log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
	return log

def train_survTrace():
	STConfig['data'] = 'metabric'
	df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)
	model = SurvTraceSingle(STConfig)
	# execute training
	trainer = Trainer(model)
	train_loss, val_loss = trainer.fit((df_train, df_y_train), (df_val, df_y_val))
	return train_loss, val_loss

def plot_train_val_loss(train_loss, val_loss, title):
	plt.title(title)
	plt.plot(train_loss, label='train')
	plt.plot(val_loss, label='val')
	plt.legend(fontsize=20)
	plt.xlabel('epoch',fontsize=20)
	plt.ylabel('loss', fontsize=20)
	plt.show()


# Metabric, DeepHit
train, val, test, labtrans = preprocess_metabric()
deephit_log = train_deephit(train, val, labtrans).to_pandas()

train_loss = deephit_log['train_loss']
val_loss = deephit_log['val_loss']
plot_train_val_loss(train_loss, val_loss, 'DeepHit on metabric')


# Metabric, SurvTrace
train_loss, val_loss = train_survTrace()
plot_train_val_loss(train_loss, val_loss, 'SurvTRACE on metabric')


# SUPPORT, DeepHit









