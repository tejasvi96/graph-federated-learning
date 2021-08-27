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
from papfl import *
from gapfl import *
from apfl import *
from fedavg import *

def get_data_lists(options):
    data_list=options['data_lists']
    train_data=[]
    val_data=[]
    for ind in range(len(data_list)):
        options['run_name']=data_list[ind][0]
        train_sents,val_sents=data_preprocess(options,name=data_list[ind][0],match=data_list[ind][1],jumbled=options['jumbled_run_flags'][ind])
        train_data.append(train_sents)
        val_data.append(val_sents)
    return train_data,val_data
@hydra.main(config_path="./",config_name="config_fed.yaml")
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
    options['degree']=cfg.main.degree
    options['steps_for_validation']=cfg.main.steps_for_validation
    options['algorithm']=fed_algorithm
    options['data_dir']=cfg.main.data_dir
    options['save_model']=cfg.main.save_model
    options['adaptive_alpha']=cfg.main.adaptive_alpha
    # options['is_jumbled']=cfg.main.is_jumbled
    options['expt_name']=cfg.main.expt_name
    options['jumbled']=cfg.main.jumbled
    folder_path=options['master_path']+os.sep+options['expt_name']
    if not os.path.isdir(folder_path):
      os.mkdir(folder_path)
    logger.add(folder_path+os.sep+"logs.log")
    options['folder_path']=folder_path
    options['data_lists']=[(i,None) for i in cfg.main.run_name.split()]
    options['jumbled_run_flags']=[int(i) for i in cfg.main.jumbled_run_flags.split()]
    # print(options['jumbled_run_flags'])
    logger.info(options['data_lists'])
    logger.info(options['jumbled_run_flags'])
#     data_list=[('supreme',None),('movie',None),('word_pred',None),('Taskmaster',None),('euro',None)]
    
    options['device_id']=cfg.main.device_id
    if torch.cuda.is_available:
        options['device']=torch.device("cuda")
    else:
        options['device']=torch.device("cpu")
    if torch.cuda.is_available and options['device_id']!=-1:
        options['device']=torch.device("cuda:"+str(options['device_id']))
    logger.info(options['device'])
    # options['device']='cpu'
    logger.info(options['device'])
    if do_env_setup:
        env_setup(master_path,vocab_file)
    else:
        print("Not setting environment")
        
    if do_training:
        training(fed_algorithm,params,options,master_path,options['data_lists'])
    else:
        print("Noo training")
    plot_run_metrics(options)

def training(fed_algorithm,params,options,master_path,data_lists):
    file_path=master_path+'embedding_class_corrected.pt'
    picklefile = open(file_path, 'rb')
    eng_obj=pickle.load(picklefile)
    options['vocab_size']=eng_obj.n_words
    train_data=[]
    val_data=[]
    train_data,val_data=get_data_lists(options)
    prev_loss=0
    lossValues=[]
    lossValueslocal=[]
    # options['fed_algorithm']='gapfl'
    algo_list=[alg for alg in fed_algorithm.split()]
    for fed_algorithm in algo_list:
      options['algorithm']=fed_algorithm
      print(fed_algorithm)
      if fed_algorithm == 'fedavg':
          Algo=fedavg(params,options,train_data,val_data,eng_obj,data_lists)
          lossValues,val_loss=Algo.model_train()
          train_loss_array=np.array(lossValues)
          val_loss_array=np.array(val_loss)
      elif fed_algorithm == 'gapfl':
          Algo=gapfl(params,options,train_data,val_data,eng_obj,params['init_alpha'],data_lists)
          lossValues,val_loss=Algo.model_train()
          train_loss_array=np.array(lossValues)
          val_loss_array=np.array(val_loss)
          Algo.plot_metrics()
      # fed_algorithm='apfl'
      elif fed_algorithm == 'apfl':
          Algo=apfl(params,options,train_data,val_data,eng_obj,params['init_alpha'],data_lists)
          lossValues,val_loss=Algo.model_train()
          train_loss_array=np.array(lossValues)
          val_loss_array=np.array(val_loss)
          Algo.plot_metrics()
      # fed_algorithm='papfl'
      elif fed_algorithm == 'papfl':
          Algo=papfl(params,options,train_data,val_data,eng_obj,params['init_alpha'],data_lists)
          lossValues,val_loss=Algo.model_train()
          train_loss_array=np.array(lossValues)
          Algo.plot_metrics()
          val_loss_array=np.array(val_loss)
      elif fed_algorithm == 'independent':
          lossValues,val_loss=train_independently(options,train_data,val_data,eng_obj)
          val_loss_array=np.array(val_loss)
          val_loss_array=val_loss_array.sum(axis=0)/val_loss_array.shape[0]
          
          train_loss_array=np.array(lossValues)
          train_loss_array=train_loss_array.sum(axis=0)/train_loss_array.shape[0]
          logger.info('independently')
      else:
          logger.info("Not implemented algo")
          val_loss_array=train_loss_array=None
      # logger.info(train_loss_array)
      # logger.info(val_loss_array)
def plot_run_metrics(options):
  algo_list=[alg for alg in options['algorithm'].split()]
  type_list=['val_loss','global_loss','local_loss']
  output_files=[f for f in os.listdir(options['folder_path']) if f.endswith(".hdf5")]
  for type in type_list:
    plt.figure()
    for file in output_files:
      # print(file)
      # ind1=file.find("_")
      ind2=file.find("_")
      algo=file[:ind2]
      run_type=file[ind2+1:].replace(".hdf5","")
      # print("run_type")
      # print(run_type)
      # print("algo")
      # print(algo)
      if run_type!=type:
        continue
      fp=h5py.File(options['folder_path']+os.sep+file,'r')
      data=np.array(fp.get(run_type)[:] )
      x=[i for i in range(data.shape[0])]      
      if 'val' in run_type:
        data=np.sum(data,axis=1)/data.shape[1]
        x=[i*10 for i in x] 
      plt.plot(x,data,label=algo)
    plt.legend()
    plt.title(algo_list)
    # plt.show()
    plt.savefig(options['folder_path']+os.sep+type+".png")


if __name__=="__main__":
    configsetters()