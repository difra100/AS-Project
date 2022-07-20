import os
import torch
#os.environ['TORCH'] = torch.__version__
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric import seed_everything

random_seed=280085
seed_everything(random_seed)


class GraphSage(nn.Module):
  ''' This class is used for all the architectures described in the research, all the details are described in the paper '''
  def __init__(self, n_layers):
    super(GraphSage, self).__init__()
    self.nl = n_layers
    self.SG1 = SAGEConv(2613,256, aggr = 'mean', normalize = True, bias = True)
    if self.nl > 1 and self.nl<=3:
      self.SG2 = SAGEConv(256,256, aggr = 'mean', normalize = True, bias = True)
    if self.nl > 2:
      self.SG3 = SAGEConv(256,256, aggr = 'mean', normalize = True, bias = True)

    self.FC1 = nn.Linear(256,256)
    self.FC2 = nn.Linear(256,256)
    self.FC3 = nn.Linear(256,100)
    

  
  def forward(self, x, edges):

    x = F.elu(self.SG1(x,edges))
    if self.nl > 1 and self.nl <= 3:
      x = F.elu(self.SG2(x,edges))
    if self.nl > 2:
      x = F.elu(self.SG3(x,edges))

    x = F.elu(self.FC1(x))
    x = F.elu(self.FC2(x))
    x = F.elu(self.FC3(x))


    return x

class GAT1(nn.Module): #Ex Conf4
  ''' This architecture is one of the 2 GAT networks, its aim is to be improved in order to see how it performs on the artist similarity task '''
  def __init__(self):
    super(GAT1, self).__init__()

    n_heads = 1

    self.GAT1 = GATConv(2613,256, heads = n_heads, bias = True)
    self.GAT2 = GATConv(n_heads*256,256, heads = n_heads, bias = True)

    self.l1 = nn.Linear(n_heads*256,256)
    self.l2 = nn.Linear(256,256)


    ''' The Batch normalization layers have been introduced to speed the training up, and indeed to obtain better results. '''
    self.batch1 = torch.nn.BatchNorm1d(256)
    self.batch2 = torch.nn.BatchNorm1d(256)

  
  def forward(self, x, edges):

    x = F.elu(self.GAT1(x,edges))
    x = F.elu(self.GAT2(x,edges))

    x = F.elu(self.batch1(self.l1(x)))
    x = F.elu(self.batch2(self.l2(x)))

    return x

class GAT2(nn.Module):
  ''' This architecture is one of the 2 GAT networks, its aim is to be improved in order to see how it performs on the artist similarity task '''
  def __init__(self):
    super(GAT2, self).__init__()

    n_heads = 1
    ''' The Batch normalization layers have been introduced to speed the training up, and indeed to obtain better results. '''
    self.batch1 = torch.nn.BatchNorm1d(256)
    self.batch2 = torch.nn.BatchNorm1d(256)
    self.batch3 = torch.nn.BatchNorm1d(256)


    self.GAT2_1 = GATConv(256,256, heads = n_heads, bias = True)
    self.GAT2_2 = GATConv(n_heads*256,256, heads = n_heads, bias = True)

    self.linear1 = nn.Linear(2613,256)
    self.linear2 = nn.Linear(256,256)
    self.linear3 = nn.Linear(256,256)


  
  def forward(self, x, edges):

    x = self.batch1(self.linear1(x))
    x = F.elu(x)
    x = self.batch2(self.linear2(x))
    x = F.elu(x)
    x = self.batch3(self.linear3(x))
    x = F.elu(x)

    x = F.elu(self.GAT2_1(x,edges))
    x = F.elu(self.GAT2_2(x,edges))
    

    return x

