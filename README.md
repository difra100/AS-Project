# Artist Similarity with Graph Attention Networks
In this repository is contained a re-implementation of the experiments conducted in the paper  ['Artist Similarity with Graph Neural Network'](https://archives.ismir.net/ismir2021/paper/000043.pdf).  
In addition to the main experiments are also performed further tests with the GAT layers, to test their permormances w.r.t. the already tested GraphSAGE.  
## Description of the repository

* models: That is a folder that contains the 5 trained model. Such as the one, two and three layers GraphSAGE, and the 2 GATs architectures that we explored to perform the experiments.
* instance: This file contains a torch tensor with the node attributes.
* adjacency and adjacencyCOO: They are still torch tensors that both represent the adjacency matrix of the graph, but with different format (sparse and coordinate).
* utils.py: This file contains a few of functions that were useful to load, save files.
* architectures.py: This file contains all the architectures used in the experiments.
* diz_of_artist.json: This file contains the name of the artists w.r.t. their position in the dataset.
* Train_Artist_similarity.ipynb: It is a notebook that contain the code for the training, and it also contains some graphical representation of the main results.
* Test_Artist_similarity.ipynb:  This is the second notebook in this repo, it shows how the models can be used as a recommending system, and how the flexibility of this approach can be used to generate embedding containing also non-existing artist. These experiments were not specified in the main research, indeed they are some additional tests that we decided to carry out.
## Brief description of the task
The main aim of the experiments is to train a GNN that is able to create embeddings of artists based on their similarities. Such similarities are expressed through the topology of an undirected graph.  
The most of hyperparameters are the same used in the main research, but since it was not possible to retrieve the same identical dataset there was an evident drop in the performances, moreover it was required to repeat again the hyperparameters tuning phase.  
In our experiments with the Graph ATtention (GAT) networks we were able to outperform the results from the original research, hereafter we can see some of these results.

# Results
* These are respectively the results on the loss function, for each network, either the train and test loss. And then there are also the results for the accuracy, that in our specific case was computed with the [Normalized Discounted Cumulative Gain](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf) metric.
![image](https://drive.google.com/uc?export=view&id=1iPUJYCa944Wv08ONIl9jk9vnxOsmjYj9)
![image](https://drive.google.com/uc?export=view&id=16YvwsN3ZYukWQsE90Vqw721C7W5Jnl9W)
![image](https://drive.google.com/uc?export=view&id=1VP3AFCAxufx4AUJXyubdcMrGKafK3Gn2)
* Furthermore, with the aid of some interactive tool libraries it is really easy to generate fictitious artist by only specifyng some plausible connections with itself.
![image](https://drive.google.com/uc?export=view&id=1SNiPzg3QbgAzzUh-UdoLW3AVS4ZF9YGJ)


