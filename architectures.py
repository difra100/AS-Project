import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric import seed_everything

random_seed=280085
seed_everything(random_seed)



class FCL(nn.Module):
  ''' This network is used in the ablation study to measure the contribute of the GCN layers '''
  def __init__(self):
    super(FCL, self).__init__()
    self.FC1 = nn.Linear(2613, 256)
    self.FC2 = nn.Linear(256, 256)
    self.FC3 = nn.Linear(256, 256)

  def forward(self, x, edges):
    out = F.elu(self.FC1(x))
    out = F.elu(self.FC2(out))
    out = self.FC3(out)

    return out

class GATSYFC(nn.Module):

  def __init__(self, n_heads, n_layers):
    super(GATSYFC, self).__init__()
    self.GATSY = GATSY(n_heads, n_layers)
    self.pred = Predictor(n_heads)

  def forward(self, x, edges):

    x = self.GATSY(x, edges)
    x = self.pred(x)

    return x



class Predictor(nn.Module):
  ''' This architecture is used as head of GAT2, to predict the genres of an artist '''
  def __init__(self, n_heads):
    super(Predictor, self).__init__()
    

    self.linear1 = nn.Linear(n_heads*256,n_heads*64)
    self.linear2 = nn.Linear(n_heads*64,n_heads*64)
    self.linear3 = nn.Linear(n_heads*64,25)
    self.batch1 = torch.nn.BatchNorm1d(n_heads*64)
    self.batch2 = torch.nn.BatchNorm1d(n_heads*64)

  def forward(self, x):

    x = self.linear1(x)
    x = self.batch1(x)
    x = F.elu(x)

    x = self.linear2(x)
    x = self.batch2(x)
    x = F.elu(x)
    x = self.linear3(x)
    
    return x

class GATSY(nn.Module):
  ''' This architecture is one of the 2 GAT networks, its aim is to be improved in order to see how it performs on the artist similarity task '''
  def __init__(self, n_heads, n_layers):
    super(GATSY, self).__init__()

    ''' The Batch normalization layers have been introduced to speed the training up, and indeed to obtain better results. '''
    
    
    self.batch1 = torch.nn.BatchNorm1d(256)
    self.batch2 = torch.nn.BatchNorm1d(256)
    self.batch3 = torch.nn.BatchNorm1d(256)
    self.batch4 = torch.nn.BatchNorm1d(n_heads*256)

    self.n_layer = n_layers
    GAT_l = []
    self.GAT1 = GATConv(256,256, heads = n_heads, bias = True)
    for n in range(n_layers - 1):
      GAT_l.append(GATConv(n_heads*256,256, heads = n_heads, bias = True))

    self.GAT_l = nn.Sequential(*GAT_l)

    batch_l = []

    for n in range(n_layers - 1):
      batch_l.append(torch.nn.BatchNorm1d(n_heads*256))

    self.batch_l = nn.Sequential(*batch_l)


    self.linear1 = nn.Linear(2613,256)
    self.linear2 = nn.Linear(256,256)
    self.linear3 = nn.Linear(256,256)


  
  def forward(self, x, edges):

    x = self.linear1(x)
    x = self.batch1(x)
    x = F.elu(x)
    x = self.linear2(x)
    x = self.batch2(x)
    x = F.elu(x)
    x = self.linear3(x)
    x = self.batch3(x)
    x = F.elu(x)

    x = self.GAT1(x,edges)
    x = self.batch4(x)
    x = F.elu(x)

    for i, layer in enumerate(self.GAT_l):
      x = layer(x, edges) #+ x
      x = self.batch_l[i](x)
      x = F.elu(x)
      

    
    return x

class GraphSage(nn.Module):
  ''' This class is used for all the architectures described in the research, all the details are described in the paper '''
  def __init__(self, aggr = 'mean'):
    super(GraphSage, self).__init__()
    self.SG1 = SAGEConv(2613, 256, aggr = aggr, normalize = True, bias = True)
    self.SG2 = SAGEConv(256, 256, aggr = aggr, normalize = True, bias = True)
    self.SG3 = SAGEConv(256, 256, aggr = aggr, normalize = True, bias = True)

    self.FC1 = nn.Linear(256,256)
    self.FC2 = nn.Linear(256,256)
    self.FC3 = nn.Linear(256,100)
    
    self.batch1 = torch.nn.BatchNorm1d(256)
    self.batch2 = torch.nn.BatchNorm1d(256)
    self.batch3 = torch.nn.BatchNorm1d(256)
  
  def forward(self, x, edges):

    x = self.batch1(F.elu(self.SG1(x,edges)))
    x = self.batch2(F.elu(self.SG2(x,edges)))
    x = self.batch3(F.elu(self.SG3(x,edges)))

    x = F.elu(self.FC1(x))
    x = F.elu(self.FC2(x))
    x = self.FC3(x)

    return x


