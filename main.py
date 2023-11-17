import os
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch.optim import lr_scheduler
import random
from random import choice,randrange
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math
import time
from torch_geometric import seed_everything
from sklearn.manifold import TSNE
import pandas as pd
from pytorch_metric_learning import losses
from sklearn.metrics import f1_score

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
import argparse
import shutil
import gc
# Internal libraries 
from src.utils import *
from src.architectures import *
from data.hyperparameters import *
from src.datamodule import *
from src.train import * 

set_determinism_the_old_way(deterministic = True)

parser = argparse.ArgumentParser()


# train_mode : True and test_mode : False ---> Hyperparameter Search 
# train_mode : True and test_mode : True ---> Test Model on validation set 
# train_mode : False and test_mode : True ---> Test Model on test set


parser.add_argument("-train_mode", type = bool, default = False)
parser.add_argument("-test_mode", type = bool, default = False)
parser.add_argument("-model_name", type = str, default = '')
parser.add_argument("-wb", type = bool, default = False)



parser.add_argument("-random_feat", type = bool, default = False)
parser.add_argument("-labels", type = bool, default = False)



args = parser.parse_args()


train_mode = args.train_mode
test_mode = args.test_mode
random = args.random_feat
filt = args.labels




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device selected in this execution is: ", device)


if random == True:
  print("RANDOM MODE HAS BEEN SELECTED..........")
  X = torch.load('./data/random_instance', map_location=torch.device(device))

else:
  print("LOW-LEVEL Feat. MODE HAS BEEN SELECTED..........")
  X = torch.load('./data/instance', map_location=torch.device(device)).T      # Instance matrix

A1 = torch.load('./data/adjacency', map_location=torch.device(device)) 

A = torch.load('./data/adjacencyCOO', map_location=torch.device(device))    # Adjacency matrix in the COO format, that is that supported by torch geometric



art_of_interest = torch.load('./data/intrst_artists.pt')
labels = torch.load('./data/labels.pt')
if filt:
  X = X[art_of_interest].detach()
  A1 = A1[art_of_interest, :][:, art_of_interest].detach()
  labels = labels[art_of_interest].detach()
  A = torch.nonzero(A1).T.type(torch.LongTensor).detach()

label_diz = load_data('data/encode_labels.json') # Genre to idx
label_diz2 = {label_diz[key] : key for key in label_diz} # Idx to genre
artists = load_data('data/artist_genres.json') # Artist names
family_diz = load_data('data/family_diz.json')


 
num_classes = torch.unique(labels).shape[0]


num_samples = X.shape[0]
print('The number of samples and classes is {} and {} respectively'.format(num_samples, num_classes))


num2artist = load_data('data/dizofartist.json')
artist2num = {num2artist[key]:key for key in num2artist}

data_for_train_ = int(len(art_of_interest)*filt_perc_train) if filt == True else n_of_vtrain
data_for_train = int(len(art_of_interest)*filt_perc_train) + int(len(art_of_interest)*filt_perc_test) if filt == True else n_of_train
fin_val = len(art_of_interest) if filt == True else n_of_test 


''' This variable contains the indices for the splitting, that are necessary to compute the masks, according to the torch geometric pipeline '''
data_summary = {'train_with_val' : {'low' : 0, 'high': data_for_train_},
                'train' : {'low' : 0, 'high' : data_for_train},
                'val' : {'low' : data_for_train_, 'high' : data_for_train},
                'test' : {'low' : data_for_train, 'high' : fin_val}}



total_mask = torch.zeros(X.shape[0], dtype = torch.bool)


vtrain_mask = total_mask.clone()
train_mask = total_mask.clone()
val_mask = total_mask.clone()
test_mask = total_mask.clone()
eval_val = total_mask.clone()


vtrain_mask[data_summary['train_with_val']['low']:data_summary['train_with_val']['high']] = True
val_mask[data_summary['val']['low']:data_summary['val']['high']] = 1
train_mask[data_summary['train']['low']:data_summary['train']['high']] = 1
test_mask[data_summary['test']['low']:data_summary['test']['high']] = 1

eval_val[data_summary['train_with_val']['low']:data_summary['val']['high']] = 1

kwargs = {'vtrain_mask':vtrain_mask, 'train_mask':train_mask, 'val_mask':val_mask, 'test_mask':test_mask}



data = Data(x=X.to(device), edge_index = A.to(device), y = labels.to(device), **kwargs)


class data_split:
  ''' This class shows an alternative to the torch geometric masks procedure, it was necessary at inference time, where was needed the whole graph for the embedding compuutation '''
  def __init__(self, data, low, high):
    ''' Starting from the index 0 to 11260, we choose the interval of intersting samples
        self.data: contains the whole dataset (nodes and edges)
        self.rang: define the boundaries
        self.get_split_from_bounds perform the splitting, returning a x and edge_index attribute resembling the torch geometric Data objects.'''
    self.data = data
    self.rang = torch.arange(low, high + 1,1, device = device)
    self.get_split_from_bounds(low, high)

  def get_split_from_bounds(self, low, high):
    self.x= self.data.x[low:high]
    v1_0 = self.data.edge_index[0]
    v2_0 = self.data.edge_index[1]
    v1_1 = v1_0[v1_0 < high]
    v1_2 = v1_1[v1_1 >= low]

    v2_1 = v2_0[v1_0 < high]
    v2_2 = v2_1[v1_1 >= low]
    v2_3 = v2_2[v2_2 < high]
    v2_4 = v2_3[v2_3 >= low]
    v1_3 = v1_2[v2_2 < high]
    v1_4 = v1_3[v2_3 >= low]

    self.edge_index = torch.cat((v1_4.unsqueeze(0), v2_4.unsqueeze(0)), dim = 0)
    
    return self.x, self.edge_index

  def split_for_inference(self, low_train, low_test, high_train, high_test):
    ''' At inference time we need to compute the embedding through the train and test artists, but we don want to consider the linkings between the test artist, those must be predicted.
        This function takes as input the boundaries of the train, and test set, computes the edge indices by removing the undesired connection.
        This method will be used to compute the accuracy. '''
    
    final_edge_indices = torch.tensor([[],[]], device = device)
    for edge in range(self.edge_index.shape[1]):
      up = self.edge_index[0][edge].item()
      down = self.edge_index[1][edge].item()

      if up in range(low_test,high_test) and down in range(low_test, high_test): # If the connection is between test artist we remove it from the edge indices.
        continue

      else:
        final_edge_indices = torch.cat((final_edge_indices, self.edge_index[:,edge].reshape((2,1))), dim = 1)

    if device.type == 'cuda':
      return self.x, final_edge_indices.type(torch.cuda.LongTensor)
    else:
      return self.x, final_edge_indices.type(torch.LongTensor)





if (train_mode == True and test_mode == False) or ((train_mode == True and test_mode == True)):
  trainmask = data.vtrain_mask
  testmask = data.val_mask
  mode = 1
  if ((train_mode == True and test_mode == True)): 
    mode = 3
  
elif train_mode == False and test_mode == True:
  trainmask = data.train_mask
  testmask = data.test_mask
  mode = 2

train_loader = NeighborLoader(data, input_nodes = trainmask, num_neighbors=[n_of_neigh]*n_layers, shuffle = True, batch_size = batch_size)
test_loader = NeighborLoader(data, input_nodes = testmask, num_neighbors=[n_of_neigh]*n_layers, shuffle = False, batch_size = batch_size)

mode_diz = {1: "You have enabled hyperparameter tuning modality.........",
            2: "You have enabled test set evaluation.........",
            3: "You have enabled val set evaluation...."}

print(mode_diz[mode])

datamodule = pl_Dataset_(train_loader, test_loader)

datamodule.setup("fit")
datamodule.setup("test")

prefix = "./Sweeps/"
exp_name = "prova"
tot_dir = prefix + exp_name + '/'

checkpoint_callback = ModelCheckpoint(dirpath = tot_dir,
        save_top_k=10, monitor="Loss on test (UNSPV)", mode="min")

early_stop = EarlyStopping(monitor='Loss on test (UNSPV)', patience=5, mode="min")

model = GATSYFC(n_heads, n_layers)

pl_training_MDL = TrainingModule(model, lr, wd, num_epochs, loss_mode, n_heads)

num_gpu = 1 if torch.cuda.is_available() else 0

# if wb:
#     # initialise the wandb logger and name your wandb project
#     wandb_logger = WandbLogger(project=project_name, name = "exp_name", config = hyperparameters, entity = 'small_sunbirds')

#     trainer = pl.Trainer(
#         max_epochs = hyperparameters['epochs'],  # maximum number of epochs.
#         gpus=num_gpu,  # the number of gpus we have at our disposal.
#         default_root_dir=prefix, logger = wandb_logger, callbacks = [Get_Metrics(), early_stop, checkpoint_callback], enable_checkpointing = True
#     )

trainer = pl.Trainer(
        max_epochs = num_epochs,  # maximum number of epochs.
        gpus=num_gpu,  # the number of gpus we have at our disposal.
        default_root_dir=prefix, callbacks = [Get_Metrics(), early_stop, checkpoint_callback], enable_checkpointing = True
    )

trainer.fit(model = pl_training_MDL, datamodule = datamodule)

print("Best model path is:", checkpoint_callback.best_model_path)
# and prints it score
print("Best model score is:\n", checkpoint_callback.best_model_score)




model = GATSYFC(n_heads, n_layers)


test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path, model = model, lr = lr, wd = wd, num_epochs = num_epochs, loss_mode = loss_mode, n_heads = n_heads)

trainer.test(test_model, dataloaders=datamodule.val_dataloader())









