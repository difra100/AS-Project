import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance
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
from utils import *
from sklearn.metrics import f1_score
import wandb

import argparse
import shutil
import os
import gc
# Internal libraries 
from src.utils import *
from src.architectures import *

set_determinism_the_old_way(deterministic = True)

parser = argparse.ArgumentParser()

parser.add_argument("-train_mode", type = bool, default = False)
parser.add_argument("-test_mode", type = bool, default = False)
parser.add_argument("-random_feat", type = bool, default = False)

parser.add_argument("-model_name", type = str, default = '')
parser.add_argument("-wb", type = bool, default = False)






args = parser.parse_args()


train_mode = args.train_mode
test_mode = args.test_mode
random = args.random_feat



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device selected in this execution is: ", device)