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
import math
import time
from torch_geometric import seed_everything
from sklearn.manifold import TSNE
import pprint


from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
import wandb
import argparse

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
parser.add_argument("-resume_sweep", type = str, default = '')
parser.add_argument("-count", type = int, default = 220)




parser.add_argument("-random_feat", type = bool, default = False)
parser.add_argument("-labels", type = bool, default = False)



args = parser.parse_args()


train_mode = args.train_mode
test_mode = args.test_mode
random = args.random_feat
filt = args.labels
wb = args.wb
model_name = args.model_name
resume_sweep = args.resume_sweep
count = args.count

if model_name == 'SAGE':
  n_heads = None

if filt:
  loss_mode = 'triplet+'
else:
  loss_mode = 'triplet'




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



if filt:
  X = X[art_of_interest].detach()
  A1 = A1[art_of_interest, :][:, art_of_interest].detach()
  labels = labels[art_of_interest].detach()
  A = torch.nonzero(A1).T.type(torch.LongTensor).detach()
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


def sweep_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, resume = True if resume_sweep != '' else False):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        if model_name == 'SAGE':
          print("GRAPHSAGE SWEEP TOOL ENABLED")
          metrics = train_sweep(seed, train_loader, test_loader, None, 3, config.lr, config.wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)
        else:
          print("GATSY SWEEP TOOL ENABLED")
          metrics = train_sweep(seed, train_loader, test_loader, config.n_heads, config.n_layers, config.lr, config.wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)
        
        if loss_mode == 'triplet+':
          accuracy_array = np.array(metrics[0])
          mean_acc, std_acc = np.mean(accuracy_array), np.std(accuracy_array)
          print("MEAN AND STANDARD DEVIATION: ", mean_acc, std_acc)
          wandb.log({"nDCG on test (Mean)": mean_acc, "nDCG on test (Std.)": std_acc})

          f1s_array = np.array(metrics[1])
          mean_f1s, std_f1s = np.mean(f1s_array), np.std(f1s_array)
          print("MEAN AND STANDARD DEVIATION: ", mean_f1s, std_f1s)
          wandb.log({"f1-score on test (Mean)": mean_f1s, "f1-score on test (Std.)": std_f1s})
        else:
          accuracy_array = np.array(metrics[0])
          mean_acc, std_acc = np.mean(accuracy_array), np.std(accuracy_array)
          print("MEAN AND STANDARD DEVIATION: ", mean_acc, std_acc)
          wandb.log({"nDCG on test (Mean)": mean_acc, "nDCG on test (Std.)": std_acc})



if mode != 1:
  train_generic(seed, train_loader, test_loader, n_heads, n_layers, lr, wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)


else:
  if n_heads != None:
    sweep_config['parameters'] = parameters_dict_GAT
  else: 
    sweep_config['parameters'] = parameters_dict_SAGE
  if resume_sweep != '':
      sweep_id = resume_sweep
      print("RESUMED PAST SWEEP....")
  else:
      sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity_name)
  pprint.pprint(sweep_config)

    

  wandb.agent(sweep_id, sweep_train, count = count, project = project_name, entity= entity_name)




if wb:
  wandb.finish()










