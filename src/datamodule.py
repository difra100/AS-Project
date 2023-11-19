import pytorch_lightning as pl

from src.utils import *
from data.hyperparameters import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


art_of_interest = torch.load('./data/intrst_artists.pt')
labels = torch.load('./data/labels.pt')

label_diz = load_data('data/encode_labels.json') # Genre to idx
label_diz2 = {label_diz[key] : key for key in label_diz} # Idx to genre
artists = load_data('data/artist_genres.json') # Artist names
family_diz = load_data('data/family_diz.json')


 
num_classes = torch.unique(labels).shape[0]


# num_samples = X.shape[0]
# print('The number of samples and classes is {} and {} respectively'.format(num_samples, num_classes))


num2artist = load_data('data/dizofartist.json')
artist2num = {num2artist[key]:key for key in num2artist}




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




class pl_Dataset_(pl.LightningDataModule):

    def __init__(self, train_loader, val_loader):

        self.train_loader = train_loader
        self.val_loader = val_loader


        

    def setup(self, stage=None):
        if stage == 'fit':
            print("")
        elif stage == 'test':
            print("")
            
    def train_dataloader(self, *args, **kwargs):
        return self.train_loader
    def val_dataloader(self, *args, **kwargs):
        return self.val_loader
    