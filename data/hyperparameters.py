lamb = 0.8 # This is the value used for the distance weighted sampling
n_of_neigh = 25 # this is the number of neighbors to fix the computational effort
n_of_vtrain = 9022 # This is the number of val-train artists
n_of_train = 10190 # This is the number of train artists
n_of_test = 11261 # This is the total number of artists
n_heads = 1
lr = 6e-5
wd = 0.1
n_layers = 2
batch_size = 512
num_epochs = 100
filt_perc_train = 0.8
filt_perc_test = 0.1
K = 200
seed = 42

if n_heads != None:
    config = {
        "n_heads" : n_heads,
        "lr" : lr,
        "wd" : wd,
        "n_layers" : n_layers
    }
else:
    config = {
        "lr" : lr,
        "wd" : wd
    }

sweep_config = {
    'method': 'random'
}
sweep_config['metric'] = {'name': 'nDCG@200 on test',
                          'goal': 'maximize'
                         }

parameters_dict_GAT = {
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'wd': {
        'values': [0, 1e-6, 1e-4, 1e-2]
    },
    'n_heads': {
        'values': [1, 2, 4, 6]
    }, 
    'n_layers': {
        'values': [1, 2, 3, 4]
    }
    
}
parameters_dict_SAGE = {
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'wd': {
        'values': [0, 1e-6, 1e-4, 1e-2]
    },
    'n_heads': {
        'values': [None]
    }
    
}

project_name = "GATSY Graph Attention Network for Music Artist Similarity"
entity_name = 'small_sunbirds'
