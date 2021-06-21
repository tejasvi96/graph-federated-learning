# graph-federated-learning
Implementation  of graph based adaptive federated learning , federated averaging on the conversational datasets.
The problem of next word prediction is considered.
![This is a alt text.](/FL_problem.png " Text Prediction ")

The project is a part of my Master's thesis work. We consider the problem of Next word prediction in the setup of federated learning here. General FL approaches learning a common centralized model for a task which performs well in IID case. Some modifications work on the personalization aspect by keeping a separate client copy which aims to take the best of the global world as well as the local world. The approaches make use of the client server interaction only. 

In this project we make use of the client - client interaction in the FL setting for the next word prediction problem. Our work is based on the hypothesis that those who know each other or chat each other they are willing to share their data though not directly but there model. 
We assume that there is no data sharing and only the model parameters that are shared among the neighbours. We extend the existing [APFL](https://arxiv.org/pdf/2003.13461.pdf) algorithm to incorporate the client client interaction in such a setup. The mixing parameter is now defined between two clients which can be compared to a notion of edge weights in a graph. Thus going on the lines of apfl, we maintain 3 models at each client local copy of global model, private model , personalized model. The personalized model for each client is a convex combination of its own private model and its neighbours sharable global model. Through the use of the mixing parameter we aim to capture the correlation between the textual similarity and the learnt mixture weights through the algorithm.

The setup looks like this -

![This is a alt text.](/graph.png " Setup of Graph")


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

```python Language_modelling_inference_colab.py -s ' How re   ' -l 2 -p 3 -t 'euro word_pred supreme Taskmaster movie'  ```
where -s parameter specifies the sentence, l parameter sets the sentence length, p is for the number of words to predict, t is a string of space separated datasets on which the personalized models have been trained. 

To compare the similarity of the two corpuses, set the dataset1 and datasets 2 lists in the language_model_similarity.py file. The same should be defined in the get_sents function in the Language_model_training_colab.py file. Run

``` python language_model_similarity.py ``` to get similarity scores for two corpuses using the USE(Universal Sentence Encoder) architecture.

