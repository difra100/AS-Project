import torch
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError
from torch_geometric.utils import structured_negative_sampling
from torchmetrics.functional import pairwise_euclidean_distance

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.architectures import *
from data.hyperparameters import *


class Get_Metrics(Callback):

  def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

    
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

        f1_s = self.compute_f1_score(pl_module.f1_list_true, pl_module.f1_list_pred)
        pl_module.f1_list_true = []
        pl_module.f1_list_pred = []





class TrainingModule(pl.LightningModule):

    def __init__(self, model, lr, wd, num_epochs, loss_mode, n_heads):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.num_epochs = num_epochs
        self.loss_mode = loss_mode
        self.n_heads = n_heads
        self.model = model
        
        self.loss = torch.nn.TripletMarginLoss(margin=0.2)
        if self.loss_mode == 'triplet+':
            self.loss_predictor = nn.CrossEntropyLoss()
            self.predictor = Predictor(self.n_heads)

        
            self.f1_list_pred = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')
            self.f1_list_true = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')
        # Metric lists
        self.loss_train_unspv = []
        self.loss_train_spv = []

        self.loss_test_unspv = []
        self.loss_test_spv = []



    def training_step(self, batch, batch_idx):
        
        out = self.model(batch.x, batch.edge_index)
        print("1: ", out.shape)

        out_l = out[:batch.batch_size]

        if self.loss_mode.startswith('triplet'):
          positives, negatives = get_triplets(out.clone(), batch.edge_index.clone(), batch.batch_size, get_vecs = True, label_idx = batch.y if self.loss_mode == 'triplet+' else None)
          
          loss_train = self.loss(out_l, positives, negatives)
          
          #Add the loss term in the train
          self.loss_train_unspv.append(loss_train)
          
          if self.loss_mode == 'triplet+':
            print("1: ", out_l.shape)

            out_n = self.predictor(out_l)
            
            loss2 = self.loss2(out_n, batch.y[:batch.batch_size]) 

            # Add the loss term in the train
            self.loss_train_spv.append(loss2)

            loss_train +=  loss2
        
        return loss_train

    def validation_step(self, batch, batch_idx):
        print("Testing step....")
        
        

        out = self.model(batch.x, batch.edge_index)
        out_l = out[:batch.batch_size]

        if self.loss_mode.startswith('triplet'):
            positives, negatives = get_triplets(out.clone(), batch.edge_index.clone(), batch.batch_size, get_vecs = True, label_idx = batch.y if self.loss_mode == 'triplet+' else None)
            loss_test = self.loss(out_l, positives, negatives)
            self.loss_test_unspv.append(loss_test)
            print("2: ", out_l.shape)

            if self.loss_mode == 'triplet+':
                print("2: ", out_l.shape)

                out_n = self.predictor(out_l)
                loss2 = self.loss2(out_n, batch.y[:batch.batch_size])
                self.loss_test_spv.append(loss2)

            
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
        
        if self.loss_mode == 'triplet+':
            out_n = self.predictor(out_l)
            loss2 = self.loss2(out_n, batch.y[:batch.batch_size])
            loss_pred.append(loss2)
        # f1_s = self.compute_f1_score(batch.y[:batch.batch_size], out_n)
        self.f1_list_pred = torch.cat([self.f1_list_pred, torch.argmax(F.softmax(out_n, dim = -1), dim = -1)])
        self.f1_list_true = torch.cat([self.f1_list_true, batch.y[:batch.batch_size]])


        return loss

    def configure_optimizers(self):
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min= 0, last_epoch= -1, verbose=True)
        if self.loss_mode == 'triplet':
            return [self.optimizer], [self.scheduler]
        else:
            self.pred_opt = torch.optim.Adam(self.predictor.parameters(), lr = self.lr, weight_decay = self.wd)
            self.scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingLR(self.pred_opt, T_max=self.num_epochs, eta_min= 0, last_epoch= -1, verbose=True)
            return [self.optimizer, self.pred_opt], [self.scheduler, self.scheduler_pred]


def compute_f1_score(true, pred, avg = 'macro'):

    pred = pred.detach().cpu().numpy()
    true = true.cpu().numpy()

    f1_s = f1_score(true, pred, labels = np.arange(num_classes), average = avg)

    return f1_s

def eval_accuracy(model, data_summary, mode = 1): 
    ''' This function computes the Normalized Discounted Cumulative Gain, that is the metric adopted in the research. '''
    if mode == 1:
      low_train = data_summary['train_with_val']['low']
      high_train = data_summary['train_with_val']['high']
      low_test = data_summary['val']['low']
      high_test = data_summary['val']['high']

    else:
      low_train = data_summary['train']['low']
      high_train = data_summary['train']['high']
      low_test = data_summary['test']['low']
      high_test = data_summary['test']['high']      


    
    ''' It is necessary to compute the embedding by condisering the edges between train and test data, but without considering the linkings between test samples, because they ,must be predicted in the evaluation. '''
    inference_data = data_split(data, low = low_train, high = high_test).split_for_inference(low_train, low_test, high_train, high_test) # This function takes care of the link remotion.
    out = self.model(inference_data[0], inference_data[1])[torch.arange(low_test,high_test)]

    A_acc = A1[low_test:high_test, low_test:high_test]

    print("Evaluation step....")
    test_embs = out.to(torch.device('cpu')).numpy()
    
    neigh=NearestNeighbors(n_neighbors=(K+1),algorithm='ball_tree').fit(test_embs) #With the K-NN we get the nearest 
    
    dist,ind=neigh.kneighbors(test_embs) 

    acc=[]

    c=0
    for test_artist in range(high_test-low_test):
        summ=0
        # ideal=len([i for i in range(self.test[0],self.test[0]+A_acc[k,:].shape[0]) if A_acc[k,i]!=0]) 
        
        ideal = A_acc[ind[c][0]].sum().item()
        
        
        
        den = self.calcG(ideal)
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


