import torch
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError
from torch_geometric.utils import structured_negative_sampling
from torchmetrics.functional import pairwise_euclidean_distance
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import gc
import shutil
import os
import pandas as pd
from pytorch_metric_learning import losses
from sklearn.metrics import f1_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math


from src.architectures import *
from src.utils import *
from src.datamodule import *
from data.hyperparameters import *




def train_sweep(seed, train_loader, test_loader, n_heads, n_layers, lr, wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = False):
  metrics = [[], []]
  seeds_list = [43, 1337, 7, 777, 9876, 54321, 123456, 999, 31415, 2022]
  for seed_ in seeds_list:
    accuracy, f1_s = train_generic(seed_, train_loader, test_loader, n_heads, n_layers, lr, wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = wb)

    metrics[0].append(accuracy)
    metrics[1].append(f1_s)

  return metrics

def train_generic(seed, train_loader, test_loader, n_heads, n_layers, lr, wd, num_epochs, loss_mode, K, data, A1, data_summary, mode, wb = False):

  set_seed(seed)
  prefix = "../Sweeps/"
  shutil.rmtree(prefix, ignore_errors=True)
  os.makedirs(prefix, exist_ok=True)
  
  datamodule = pl_Dataset_(train_loader, test_loader)

  datamodule.setup("fit")
  datamodule.setup("test")

  
  if n_heads != None:
   exp_name = f"{seed}_{n_layers}_{n_heads}_{lr}_{wd}_{loss_mode}"
   model = GATSY(n_heads, n_layers)
   if loss_mode == 'triplet+':
    predictor = Predictor(n_heads)
   else: 
    predictor = None

  else:
   exp_name = f"{seed}_{n_layers}_{lr}_{wd}_{loss_mode}"
   model = GraphSage()
   predictor = None
    
  tot_dir = prefix + exp_name + '/'

  checkpoint_callback = ModelCheckpoint(dirpath = tot_dir,
          save_top_k=1, monitor="nDCG@200 on test", mode="max")

  early_stop = EarlyStopping(monitor='Loss on test (UNSPV)', patience=5, mode="min")


  pl_training_MDL = TrainingModule(model, predictor, lr, wd, num_epochs, loss_mode, n_heads, K, data, A1, data_summary, mode)

  num_gpu = 1 if torch.cuda.is_available() else 0

  if wb:
      # initialise the wandb logger and name your wandb project
      wandb_logger = WandbLogger(project=project_name, name = exp_name, config = config, entity = entity_name)

      trainer = pl.Trainer(
          max_epochs = num_epochs,  # maximum number of epochs.
          gpus=num_gpu,  # the number of gpus we have at our disposal.
          default_root_dir=prefix, logger = wandb_logger, callbacks = [Get_Metrics(), early_stop, checkpoint_callback], enable_checkpointing = True
      )
  else:
    trainer = pl.Trainer(
            max_epochs = num_epochs,  # maximum number of epochs.
            gpus=num_gpu,  # the number of gpus we have at our disposal.
            default_root_dir=prefix, callbacks = [Get_Metrics(), early_stop, checkpoint_callback], enable_checkpointing = True
        )

  trainer.fit(model = pl_training_MDL, datamodule = datamodule)

  print("Best model path is:", checkpoint_callback.best_model_path)
  # and prints it score
  print("Best model score is:\n", checkpoint_callback.best_model_score)


  if n_heads != None:
    model = GATSY(n_heads, n_layers)
    if loss_mode == 'triplet+':
      predictor = Predictor(n_heads)
    else: 
      predictor = None
  else:
     model = GraphSage()
     predictor = None


  test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path, model = model, predictor = predictor, lr = lr, wd = wd, num_epochs = num_epochs, loss_mode = loss_mode, n_heads = n_heads, K = K, data = data, A1 = A1, data_summary= data_summary, mode = mode)

  trainer.test(test_model, dataloaders=datamodule.val_dataloader())


  # loss_test = sum(test_model.loss_test_unspv)/len(test_model.loss_test_unspv)
  accuracy = eval_accuracy(test_model.model, test_model.data, test_model.A1, test_model.data_summary, test_model.K, test_model.mode)
  print("Tested Model: ", accuracy)
  if loss_mode == 'triplet+':
     f1_s = compute_f1_score(test_model.f1_list_true, test_model.f1_list_pred)

  del datamodule
  del test_model
  del model
  del trainer
  del pl_training_MDL

  gc.collect()

  if loss_mode == 'triplet+':
    return accuracy, f1_s
  else:
    return accuracy, 0

class Get_Metrics(Callback):

  def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

    accuracy = eval_accuracy(pl_module.model, pl_module.data, pl_module.A1, pl_module.data_summary, pl_module.K, pl_module.mode)
    
    pl_module.log(name = 'nDCG@200 on test', value = accuracy, on_epoch = True, prog_bar = True, logger = True)

    loss_test_unspv = sum(pl_module.loss_test_unspv)/len(pl_module.loss_test_unspv)
    pl_module.log(name = 'Loss on test (UNSPV)', value = loss_test_unspv, on_epoch=True, prog_bar=True, logger=True)
    pl_module.loss_test_unspv = []

    loss_train_unspv = sum(pl_module.loss_train_unspv)/len(pl_module.loss_train_unspv)
    pl_module.log(name = 'Loss on train (UNSPV)', value = loss_train_unspv, on_epoch=True, prog_bar=True, logger=True)
    pl_module.loss_train_unspv = []

    if pl_module.loss_mode == 'triplet+':

        loss_test_spv = sum(pl_module.loss_test_spv)/len(pl_module.loss_test_spv)
        pl_module.log(name = 'Loss on test (SPV)', value = loss_test_spv, on_epoch=True, prog_bar=True, logger=True)
        pl_module.loss_test_spv = []

        loss_train_spv = sum(pl_module.loss_train_spv)/len(pl_module.loss_train_spv)
        pl_module.log(name = 'Loss on train (SPV)', value = loss_train_spv, on_epoch=True, prog_bar=True, logger=True)
        pl_module.loss_train_spv = []

        f1_s = compute_f1_score(pl_module.f1_list_true, pl_module.f1_list_pred)
        pl_module.log(name = 'f1-score on test', value = f1_s, on_epoch=True, prog_bar=True, logger=True)
        pl_module.f1_list_true = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')
        pl_module.f1_list_pred = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')





class TrainingModule(pl.LightningModule):

    def __init__(self, model, predictor, lr, wd, num_epochs, loss_mode, n_heads, K, data, A1, data_summary, mode):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.num_epochs = num_epochs
        self.loss_mode = loss_mode
        self.n_heads = n_heads
        self.model = model
        self.K = K
        self.data = data
        self.A1 = A1
        self.data_summary = data_summary
        self.mode = mode
        
        self.loss = torch.nn.TripletMarginLoss(margin=0.2)
        if self.loss_mode == 'triplet+':
            self.loss_predictor = nn.CrossEntropyLoss()

            self.predictor = predictor

        
            self.f1_list_pred = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')
            self.f1_list_true = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')
        # Metric lists
        self.loss_train_unspv = []
        self.loss_train_spv = []

        self.loss_test_unspv = []
        self.loss_test_spv = []



    def training_step(self, batch, batch_idx):
        
        out = self.model(batch.x, batch.edge_index)
        # print("1: ", out.shape)

        out_l = out[:batch.batch_size, :]
        # print("STUFF: ", out_l.shape)
        if self.loss_mode.startswith('triplet'):
          positives, negatives = get_triplets(out.clone(), batch.edge_index.clone(), batch.batch_size, get_vecs = True, label_idx = batch.y if self.loss_mode == 'triplet+' else None)
          
          loss_train = self.loss(out_l, positives, negatives)
          
          #Add the loss term in the train
          self.loss_train_unspv.append(loss_train)
          
          if self.loss_mode == 'triplet+':
            # print("1: ", out_l.shape)

            out_n = self.predictor(out_l)
            
            loss2 = self.loss_predictor(out_n, batch.y[:batch.batch_size]) 

            # Add the loss term in the train
            self.loss_train_spv.append(loss2)

            loss_train +=  loss2
        
        return loss_train

    def validation_step(self, batch, batch_idx):
        # print("Testing step....")
        
        if len(self.loss_train_unspv)== 0:
          #  print("Skip sanity check.........")
           return

        out = self.model(batch.x, batch.edge_index)
        out_l = out[:batch.batch_size]

        if self.loss_mode.startswith('triplet'):
            positives, negatives = get_triplets(out.clone(), batch.edge_index.clone(), batch.batch_size, get_vecs = True, label_idx = batch.y if self.loss_mode == 'triplet+' else None)
            loss_test = self.loss(out_l, positives, negatives)
            self.loss_test_unspv.append(loss_test)
            # print("2: ", out_l.shape)

            if self.loss_mode == 'triplet+':
                # print("2: ", out_l.shape)

                out_n = self.predictor(out_l)
                loss2 = self.loss_predictor(out_n, batch.y[:batch.batch_size])
                self.loss_test_spv.append(loss2)

                loss_test += loss2

            
                # f1_s = self.compute_f1_score(batch.y[:batch.batch_size], out_n)
                self.f1_list_pred = torch.cat([self.f1_list_pred, torch.argmax(F.softmax(out_n, dim = -1), dim = -1)])
                self.f1_list_true = torch.cat([self.f1_list_true, batch.y[:batch.batch_size]])
  

        return loss_test
    
    def test_step(self, batch, batch_idx):

        out = self.model(batch.x, batch.edge_index)
        out_l = out[:batch.batch_size]

        if self.loss_mode.startswith('triplet'):
            positives, negatives = get_triplets(out.clone(), batch.edge_index.clone(), batch.batch_size, get_vecs = True, label_idx = batch.y if self.loss_mode == 'triplet+' else None)
            loss_test = self.loss(out_l, positives, negatives)
            self.loss_test_unspv.append(loss_test)

        
        if self.loss_mode == 'triplet+':
            out_n = self.predictor(out_l)
            loss2 = self.loss_predictor(out_n, batch.y[:batch.batch_size])
            self.loss_test_spv.append(loss2)
            
        # f1_s = self.compute_f1_score(batch.y[:batch.batch_size], out_n)
            self.f1_list_pred = torch.cat([self.f1_list_pred, torch.argmax(F.softmax(out_n, dim = -1), dim = -1)])
            self.f1_list_true = torch.cat([self.f1_list_true, batch.y[:batch.batch_size]])

            loss_test += loss2


        return loss_test

    def configure_optimizers(self):
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min= 0, last_epoch= -1, verbose=True)
        return [self.optimizer], [self.scheduler]


def compute_f1_score(true, pred, avg = 'macro'):

    pred = pred.detach().cpu().numpy()
    true = true.cpu().numpy()

    f1_s = f1_score(true, pred, labels = np.arange(num_classes), average = avg)

    return f1_s

def eval_accuracy(model, data, A1, data_summary, K = 200, mode = 1): 
    ''' This function computes the Normalized Discounted Cumulative Gain, that is the metric adopted in the research. '''
    model.eval()
    with torch.no_grad():
      if mode == 1 or mode == 3:
        low_train = data_summary['train_with_val']['low']
        high_train = data_summary['train_with_val']['high']
        low_test = data_summary['val']['low']
        high_test = data_summary['val']['high']

      else:
        low_train = data_summary['train']['low']
        high_train = data_summary['train']['high']
        low_test = data_summary['test']['low']
        high_test = data_summary['test']['high']      

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      
      ''' It is necessary to compute the embedding by condisering the edges between train and test data, but without considering the linkings between test samples, because they ,must be predicted in the evaluation. '''
      inference_data = data_split(data, low = low_train, high = high_test).split_for_inference(low_train, low_test, high_train, high_test) # This function takes care of the link remotion.
      model.to(device)
      out = model(inference_data[0].to(device), inference_data[1].to(device))[torch.arange(low_test,high_test)]

      A_acc = A1[low_test:high_test, low_test:high_test]

      print("Evaluation step....")
      test_embs = out.to(torch.device('cpu')).numpy()
      
      neigh=NearestNeighbors(n_neighbors=(K+1),algorithm='ball_tree').fit(test_embs) #With the K-NN we get the nearest 
      
      dist, ind=neigh.kneighbors(test_embs) 

      acc=[]

      c=0
      for test_artist in range(high_test-low_test):
          summ=0
          # ideal=len([i for i in range(self.test[0],self.test[0]+A_acc[k,:].shape[0]) if A_acc[k,i]!=0]) 
          
          ideal = A_acc[ind[c][0]].sum().item()
          
          den = calcG(ideal, K)
          if den==0:
              c+=1
              continue  
          for j in range(len(ind[c][1:])):
              if A_acc[ind[c][0]][ind[c][1:][j]].item()!=0:
                  summ+= 1/(math.log2(1+(j+1)))
                  
              else:
                  continue
          c+=1    
          summ/=den
          acc.append(summ)
      return sum(acc)/len(acc)

def get_triplets(embedding, edges, batch_size, get_vecs = False, label_idx = None):
  ''' The loss function to minimize is  a triplet loss function, thus we need to look for positives and negatives for each sample in the batches.
      This function takes as input:
      * embedding:  the output of the GCN.
      * edges:      the edges for the mini-batch.
      * label_idx:  if class are available, this is chosen to improve the triplet selection
      * batch_size: the size for the batch.
      * get_vecs:   It is a boolean, if True the function returns the tensor of positives and negatives, otherwise only the indices are returned.  '''

  edges_n = edges.clone()
  edges[0] = edges_n[1]
  edges[1] = edges_n[0]
  
  total_triplets = structured_negative_sampling(edges)
  total_ancor = total_triplets[0]
  total_pos = total_triplets[1]
  total_negs = total_triplets[2]

 
  shape = batch_size
  positives = torch.zeros(shape)
  negatives = torch.zeros(shape)
  for ancor in range(shape):
    pos = total_pos[total_ancor == ancor] # We get the positives that are also neighbors for the anchor
    
    # pos = pos[pos < batch_size]  
    # if pos.shape[0] > n_of_neigh:
    #   pos = pos[:n_of_neigh]
    neg = total_negs[total_ancor == ancor] # We get the negatives that are also neighbors for the anchor
    # neg = neg[neg < batch_size]  

    # if neg.shape[0] > n_of_neigh:
    #   neg = neg[:n_of_neigh]
    p = 1
    n = 1

    if pos.shape[0] == 0:
      positives[ancor] == ancor
      p = 0
    
    if neg.shape[0] == 0:
      negatives[ancor] == ancor
      n = 0
    
    if p:

      pos_index = compute_idx_p(embedding, pos, ancor, label_idx)
      positives[ancor] = pos_index
    if n:
      neg_index = compute_idx_n(embedding, neg, ancor, label_idx)
      negatives[ancor] = neg_index
  
  
  if torch.cuda.is_available():
    positives = positives.type(torch.cuda.LongTensor) 
    negatives = negatives.type(torch.cuda.LongTensor)
  else:
    positives = positives.type(torch.LongTensor)
    negatives = negatives.type(torch.LongTensor)

  
  if get_vecs:
    return embedding[positives], embedding[negatives] #Return the embedding vectors

  else:
    return positives, negatives                       # Or return the list of indices


def compute_idx_p(embedding, pos, ancor, label_idx):
# This function performs the distance weighted sampling.
# We look for all positives for the anchor sample and we weight their distance from it. We choose then the 'hard positive' namely one of the furthest positives for the anchor
  diz_pos_ = {}
  #print(pos)
  
  for idx in pos:
    if label_idx != None:
      # print(ancor, idx)
      # print(label_idx.shape)
      # print(pos)

      # if family_diz[str(label_idx[ancor].item())] != family_diz[str(label_idx[idx].item())]:
      #     continue
      if label_idx[ancor].item() != label_idx[idx].item():
          continue
    diz_pos_[idx.item()] = pairwise_euclidean_distance(embedding[ancor].unsqueeze(0), embedding[idx.item()].unsqueeze(0))[0][0].item()
  
  if len(diz_pos_) == 0:
    for idx in pos:

      # print('pos: ', pos.shape)


      # print(family_diz[str(label_idx[ancor].item())], family_diz[str(label_idx[idx].item())])

      diz_pos_[idx.item()] = pairwise_euclidean_distance(embedding[ancor].unsqueeze(0), embedding[idx.item()].unsqueeze(0))[0][0].item()

  
  max_dist = max(list(diz_pos_.values())) if max(list(diz_pos_.values()))!=0 else 1e-5
  keys = list(diz_pos_.keys())

  for key in keys:
    if diz_pos_[key]/max_dist > lamb and len(diz_pos_) != 1:
      diz_pos_.pop(key)


  return max(diz_pos_,key=diz_pos_.get)

def compute_idx_n(embedding, neg, ancor, label_idx = None):
# This function performs the distance weighted sampling.
# We look for all negatives for the anchor sample and we weight their distance from it. We choose then the 'hard negative' namely one of the closest negative for the anchor 
    diz_neg_ = {}
    #print(neg)
    for idx in neg:
      if label_idx != None:
        # if family_diz[str(label_idx[ancor].item())] == family_diz[str(label_idx[idx].item())]:
        #   continue
        if label_idx[ancor].item() == label_idx[idx].item():
          continue
      diz_neg_[idx.item()] = pairwise_euclidean_distance(embedding[ancor].unsqueeze(0), embedding[idx.item()].unsqueeze(0))[0][0].item()

    if len(diz_neg_) == 0:
      diz_neg_[ancor] = 0

    max_dist = max(list(diz_neg_.values())) if max(list(diz_neg_.values()))!=0 else 1e-5
    keys = list(diz_neg_.keys())
    for key in keys:
      if diz_neg_[key]/max_dist < 1 - lamb and len(diz_neg_) != 1:
        diz_neg_.pop(key)


    return min(diz_neg_,key=diz_neg_.get)


def calcG(ID, K):  #This method is used for the evaluation of accuracy, in particular it computes the denominator
    if ID>K:       # as described in the paper.
        ID=K
    c=1
    somm=0
    while c<=ID:
        somm+=1/(math.log2(1+c))
        c+=1
    return somm