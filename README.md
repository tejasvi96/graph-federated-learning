# graph-federated-learning

Implementation  of graph based adaptive federated learning , federated averaging on the conversational datasets.
The problem of next word prediction is considered.
![This is a alt text.](/FL_problem.png " Text Prediction ")

The project is a part of my Master's thesis work. We consider the problem of Next word prediction in the setup of federated learning here. General FL approaches learning a common centralized model for a task which performs well in IID case. Some modifications work on the personalization aspect by keeping a separate client copy which aims to take the best of the global world as well as the local world. The approaches make use of the client server interaction only. 

In this project we make use of the client - client interaction in the FL setting for the next word prediction problem. Our work is based on the hypothesis that those who know each other or chat each other they are willing to share their data though not directly but there model. 
We assume that there is no data sharing and only the model parameters that are shared among the neighbours. We extend the existing [APFL](https://arxiv.org/pdf/2003.13461.pdf) algorithm to incorporate the client client interaction in such a setup. The mixing parameter is now defined between two clients which can be compared to a notion of edge weights in a graph. Thus going on the lines of apfl, we maintain 3 models at each client local copy of global model, private model , personalized model. The personalized model for each client is a convex combination of its own private model and its neighbours sharable global model. Through the use of the mixing parameter we aim to capture the correlation between the textual similarity and the learnt mixture weights through the algorithm.

The setup looks like this -

![This is a alt text.](/FL_setup.png " Setup of Graph ")

## Federated Learning

Federated Learning is a machine learning paradigm involving decentralized learning across
multiple edge devices with objective of learning a common model with the constraints of data
privacy and on device learning. Typically there is an orchestrator(server) and a set of clients
where the data resides on the clients and the orchestrator broadcasts the model parameters to all
the clients who perform the calculations independently on their local data and share the model
parameters back. The orchestrator does the aggregation of the weight parameters received from
the dierent clients. The aggregation may be a simple vanilla averaging or a weighted averaging
based on the  flavour of algorithm being used.

![This is a alt text.](/FL_1.png " Federated Learning ")

## Personalization in Federated Learning

The traditional federated learning algorithms aim to learn a common model with the assumption that the data across the devices is IID. 

![equation](https://latex.codecogs.com/gif.latex?%5Cmin%20_%7Bw%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D%7D%5Cleft%5C%7Bf%28w%29%3A%3D%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20f_%7Bi%7D%28w%29%5Cright%5C%7D)

There are different algorithms which work on the personalization aspect by solving a variant of this problem.
In pFedme , the same is modified to 

![equation](https://latex.codecogs.com/gif.latex?%5Cmin%20_%7B%5Cboldsymbol%7B%5Ctheta_%7Bi%7D%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D%7D%28%20f_%7Bi%7D%5Cleft%28%5Ctheta_%7Bi%7D%5Cright%29&plus;%5Cfrac%7B%5Clambda%7D%7B2%7D%5Cleft%5C%7C%5Ctheta_%7Bi%7D-w%5Cright%5C%7C%5E%7B2%7D%29).

![This is a alt text.](/pfedme.png " Federated Learning ")



In apfl, the personalization problem is modified to 

![equation](https://latex.codecogs.com/gif.latex?%5Cmin%20_%7B%5Cboldsymbol%7Bv%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D%7D%20f_%7Bi%7D%5Cleft%28%5Calpha_%7Bi%7D%20%5Cboldsymbol%7Bv%7D&plus;%5Cleft%281-%5Calpha_%7Bi%7D%5Cright%29%20%5Cboldsymbol%7Bw%7D%5E%7B*%7D%5Cright%29)

![This is a alt text.](/apfl_algo.png " Federated Learning ")


for  each client **i** where *v* is the private model at  client i and *w* is the global copy.

## Proposed Algorithm

To learn a personalized model for each of the client in the graph based setup we proposed, we
modify the apfl algorithm to a graph based variant(gap). The key difference is that the mixture
weights are now learnt between every client and each of its neighbours. The personalized
model for a client is also modified to be a weighted combination of its own private model and
the global model of all its neighbours. The personalized model of the client i becomes -

![equation](https://latex.codecogs.com/gif.latex?%5Coverline%7B%5Cboldsymbol%7Bv%7D%7D_%7Bi%7D%5E%7B%28t%29%7D%3D%5Csum_%7Bj%20%5Cin%20%5Bneigbors%28i%29%5D%7D%20%28%5Calpha_%7Bij%7D/d%28i%29%29%20%5Cboldsymbol%7Bv%7D_%7Bi%7D%5E%7B%28t%29%7D&plus;%5Cleft%28%281-%5Calpha_%7Bij%7D%5Cright%29/d%28i%29%29%20%5Cboldsymbol%7Bw%7D_%7Bj%7D%5E%7B%28t%29%7D)


![This is a alt text.](/FL_gapfl.PNG " Federated Learning ")


## Datasets
We consider some realworld textual datasets to simulate the real conversation across the clients. We consider [europarl corpus](https://www.statmt.org/europarl/) (european paraliamentary proceedings), [supreme court corpus](https://confluence.cornell.edu/display/llresearch/Supreme+Court+Dialogs+Corpus) (transcripts of proceedings of supreme court), [movie dialogs](https://arxiv.org/abs/1106.3077)( actual dialog lines from some movies) , [Taskmaster](https://arxiv.org/abs/1909.05358) ( chatbot data), [Word Predictions](https://www.aclweb.org/anthology/C18-2028/)(Normal wikipedia text data).

Run script ```python dataset_download.py``` to download the used datasets.

## Code

Set the configurations in config_fed.yaml.
```
main:
    dir: '/home/tejasvi/'
    vocab_file: 'google-10000-english-no-swears.txt'
    do_env_setup : 0
    do_training : 1
    fed_algorithm: 'independent' #gapfl,fedavg,independent
    NUM_CLIENTS: 5
    init_alpha: 0.5
    num_rounds: 1502
    tau: 10
    batch_size: 128
    max_seq_len: 40
    num_lstm_layers: 1
    hid_size: 300
    dropout: 0.1
    learning_rate: 0.01
    min_learning_rate: 0.0001
    weight_decay_factor: 0.5
    run_name: 'supreme word_pred Taskmaster euro movie'
    epochs: 1
    steps_for_validation: 500
    match: None
    use_cuda: 0
    device_id: 3 
    clients: 'reddit'
    alpha_lr: 1.0
    load_model: 0
    load_model_file: 0
```

We are implementing 3 algorithms namely - gapfl, fedavg, training independently(without sharing of model parameters). 
The other two algorithms we consider are [perfedavg](https://arxiv.org/abs/2002.07948), [pfedme](https://arxiv.org/abs/2006.08848) whose code implementations were taken from [here](https://github.com/CharlieDinh/pFedMe).



First of all run ``` pip install -r requirements.txt ``` to install all the dependencies. or to do the environment setup set the parameter do_env_setup in config_fed.yaml file. It will also install all the requirements. Run the ``` python main.py ``` after setting the parameter.

To run the training set the do_training parameter in config_fed.yaml and set the datasets as space separated in the run_name. To add a new dataset add a corresponding get_data function in Language_modelling_training.py file and modify the data_preprocess function to add its calling logic. Run the ```python main.py``` again after setting the parameters.

For gapfl , set the **init_alpha** and **alpha_lr** parameters to initialize the mixture weights(alpha). Currently we are assuming the fully connected graph. To manually set the weights differently , same can be done in gapfl.py file.


To do inference make use of the Language_modelling_inference_colab.py file. Same can be run as -

```python Language_modelling_inference_colab.py -s ' How re   ' -l 2 -p 3 -t 'euro word_pred supreme Taskmaster movie' -a 'fedavg' -q 1 -k 2  ```
where s parameter specifies the sentence, l parameter sets the sentence length, p is for the number of next words to predict, t is a string of space separated datasets on which the personalized models have been trained, a stands for algorithms 'fedavg','independent','gapfl', q stands for quantisation parameter on the saved model, k is the number of top k predictions to predict for a single timestep. 

To compare the similarity of the two corpuses, set the dataset1 and datasets 2 lists in the language_model_similarity.py file. The same should be defined in the get_sents function in the Language_model_training_colab.py file. Run

``` python language_model_similarity.py ``` to get similarity scores for two corpuses using the USE(Universal Sentence Encoder) architecture.

