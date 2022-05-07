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

def train_deephit(train_set, val_set, labtrans):
	df_train, df_y_train = train_set
	x_train = df_train.to_numpy(np.float32)
	y_train = (df_y_train['duration'].values, df_y_train['event'].values)

	df_val, df_y_val = val_set
	x_val = df_val.to_numpy(np.float32)
	y_val = (df_y_val['duration'].values, df_y_val['event'].values)
	val = (x_val, y_val)

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
	log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val).to_pandas()
	return log['train_loss'], log['val_loss']

def train_survTrace(STConfig, train_set, val_set):
	df_train, df_y_train = train_set
	df_val, df_y_val = val_set
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

# load metabric data
# define the setup parameters
STConfig['data'] = 'metabric'

set_random_seed(STConfig['seed'])

hparams = {
    'batch_size': 64,
    'weight_decay': 1e-4,
    'learning_rate': 1e-3,
    'epochs': 20,
}

df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)
train_set = (df_train, df_y_train)
val_set = (df_val, df_y_val)
# train_loss, val_loss = train_survTrace(STConfig, train_set, val_set)


train_loss, val_loss = train_deephit(train_set, val_set, STConfig['labtrans'])
plot_train_val_loss(train_loss, val_loss, 'DeepHit on metabric')

train_loss, val_loss = train_survTrace(STConfig, train_set, val_set)
plot_train_val_loss(train_loss, val_loss, 'SurvTRACE on metabric')









