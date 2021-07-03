import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import time
import operator
import os
import hydra
from loguru import logger

from language_modelling_setup import *
from apfl_ours import *
from fedavg import *
logger.add("logs_file.log")

def get_data_lists(options,data_list):
    data=[]
    val_data=[]
    for ind in range(len(data_list)):
        options['run_name']=data_list[ind][0]
        train_sents,val_sents=data_preprocess(options,name=data_list[ind][0],match=data_list[ind][1])
        data.append(train_sents)
        val_data.append(val_sents)
    return data,val_data
@hydra.main(config_path="config_fed.yaml")
def configsetters(cfg):
    master_path=cfg.main.dir
    vocab_file=cfg.main.vocab_file
    do_env_setup=cfg.main.do_env_setup
    do_training=cfg.main.do_training
    fed_algorithm=cfg.main.fed_algorithm
    params={}
    params['NUM_CLIENTS']=cfg.main.NUM_CLIENTS
    params['num_rounds']=cfg.main.num_rounds
    params['tau']=cfg.main.tau
    params['init_alpha']=cfg.main.init_alpha
    options={}
    options['num_lstm_layers']=cfg.main.num_lstm_layers
    options['steps_for_validation']=cfg.main.steps_for_validation
    options['hid_size']=cfg.main.hid_size
    options['learning_rate']=cfg.main.learning_rate
    options['dropout']=cfg.main.dropout
    options['weight_decay_factor']=cfg.main.weight_decay_factor
    options['min_learning_rate']=cfg.main.min_learning_rate
    options['master_path']=master_path
    options['max_seq_len']=cfg.main.max_seq_len
    options['batch_size']=cfg.main.batch_size
    options['alpha_lr']=cfg.main.alpha_lr
    options['load_model']=cfg.main.load_model
    options['load_model_file']=cfg.main.load_model_file
    options['epochs']=cfg.main.epochs
    options['algorithm']=fed_algorithm
    data_lists=[(i,None) for i in cfg.main.run_name.split()]
    print(data_lists)
#     data_list=[('supreme',None),('movie',None),('word_pred',None),('Taskmaster',None),('euro',None)]
    
    options['device_id']=cfg.main.device_id
    options['device']=torch.device("cpu")
    if torch.cuda.is_available and options['device_id']!=-1:
        options['device']=torch.device("cuda:"+str(options['device_id']))
    print(master_path)
    if do_env_setup:
        env_setup(master_path,vocab_file)
    else:
        print("Not setting environment")
        
    if do_training:
        training(fed_algorithm,params,options,master_path,data_lists)
    else:
        print("Noo training")
        
def training(fed_algorithm,params,options,master_path,data_lists):
    file_path=master_path+'/embedding_class_corrected.pt'
    picklefile = open(file_path, 'rb')
    eng_obj=pickle.load(picklefile)
    options['vocab_size']=eng_obj.n_words
    train_data=[]
    val_data=[]
    train_data,val_data=get_data_lists(options,data_lists)
  
    
    prev_loss=0
    lossValues=[]
    lossValueslocal=[]
    if fed_algorithm == 'fedavg':
        Algo=fedavg(params,options,train_data,val_data,eng_obj)
        lossValues,val_loss=Algo.model_train()
        train_loss_array=np.array(lossValues)
        val_loss_array=np.array(val_loss)
    elif fed_algorithm == 'gapfl':
        Algo=gapfl(params,options,train_data,val_data,eng_obj,params['init_alpha'],data_lists)
        lossValues,val_loss=Algo.model_train()
        train_loss_array=np.array(lossValues)
        val_loss_array=np.array(val_loss)
    elif fed_algorithm == 'independent':
        lossValues,val_loss=train_independently(options,train_data,val_data,eng_obj)
        
        val_loss_array=np.array(val_loss)
        val_loss_array=val_loss_array.sum(axis=0)/val_loss_array.shape[0]
        
        train_loss_array=np.array(lossValues)
        train_loss_array=train_loss_array.sum(axis=0)/train_loss_array.shape[0]
        print('independently')
    else:
        print("Not implemented algo")
        val_loss_array=train_loss_array=None
    print(train_loss_array)
    print(val_loss_array)
if __name__=="__main__":
    configsetters()