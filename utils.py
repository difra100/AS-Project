import json
import torch

def load_data(name):
    jfile = open(name, "r")
    dicti = json.load(jfile)
    return dicti
def save_data(dicti,name):
      jfile = open(name, "w")
      jfile = json.dump(dicti, jfile)   
    
def save_model(checkpoint, path):

  torch.save(checkpoint, path)

def load_model(path,model, device):

  checkpoint = torch.load(path, map_location=device)
  if not 'modelState2' in checkpoint:
    model.load_state_dict(checkpoint['modelState']) 
  else:
    model.GATSY.load_state_dict(checkpoint['modelState']) 
    model.pred.load_state_dict(checkpoint['modelState2']) 

     
  
  
  return checkpoint


