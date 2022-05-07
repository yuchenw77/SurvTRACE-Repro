'''SEER data comes from https://seer.cancer.gov/data/
'''
from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceMulti
from survtrace.train_utils import Trainer
from survtrace.config import STConfig
from competing_helper import CauseSpecificNet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchtuples as tt

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv

import matplotlib.pyplot as plt

def train_survTrace(STConfig, train_set, val_set):
	hparams = {
	    'batch_size': 1024,
	    'weight_decay': 0,
	    'learning_rate': 1e-4,
	    'epochs': 100,
	}
	model = SurvTraceMulti(STConfig)
	trainer = Trainer(model)
	train_loss, val_loss = trainer.fit((df_train, df_y_train), (df_val, df_y_val),
        batch_size=hparams['batch_size'],
        epochs=hparams['epochs'],
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
        val_batch_size=10000,)
	return train_loss, val_loss

def train_deepHit(labtrans, train_set, val_set):
	df_train, df_y_train = train_set
	x_train = df_train.to_numpy(np.float32)
	print(df_y_train.head())
	y_train = (df_y_train['duration'].values, df_y_train['event_0'].to_numpy(np.int_) + df_y_train['event_1'].to_numpy(np.int_) * 2)

	df_val, df_y_val = val_set
	x_val = df_val.to_numpy(np.float32)
	y_val = (df_y_val['duration'].values, df_y_val['event_0'].to_numpy(np.int_) + df_y_val['event_1'].to_numpy(np.int_) * 2)
	val = (x_val, y_val)
	in_features = x_train.shape[1]
	num_nodes_shared = [64, 64]
	num_nodes_indiv = [32]
	num_risks = 2
	out_features = len(labtrans.cuts)
	batch_norm = True
	dropout = 0.1

	net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                       out_features, batch_norm, dropout)
	optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01,
                            cycle_eta_multiplier=0.8)
	model = DeepHit(net, optimizer, alpha=0.2, sigma=0.1,
                duration_index=labtrans.cuts)
	epochs = 512
	batch_size = 256
	callbacks = [tt.callbacks.EarlyStoppingCycle()]
	verbose = True # set to True if you want printout
	log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val)
	log = log.to_pandas()
	return log['train_loss'], log['val_loss']

def plot_train_val_loss(train_loss, val_loss, title):
	plt.title(title)
	plt.plot(train_loss, label='train')
	plt.plot(val_loss, label='val')
	plt.legend(fontsize=20)
	plt.xlabel('epoch',fontsize=20)
	plt.ylabel('loss', fontsize=20)
	plt.show()



# define the setup parameters
STConfig['data'] = 'seer'
STConfig['num_hidden_layers'] = 2
STConfig['hidden_size'] = 16
STConfig['intermediate_size'] = 64
STConfig['num_attention_heads'] = 2
STConfig['initializer_range'] = .02
STConfig['early_stop_patience'] = 5
set_random_seed(STConfig['seed'])

df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)
train_set = (df_train, df_y_train)
val_set = (df_val, df_y_val)

train_loss, val_loss = train_deepHit(STConfig['labtrans'], train_set, val_set)
plot_train_val_loss(train_loss, val_loss, 'deephit on seer')

train_loss, val_loss = train_survTrace(STConfig, train_set, val_set)
plot_train_val_loss(train_loss, val_loss, 'survTrace on seer')



