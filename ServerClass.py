import itertools
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np
import h5py
from ModelClass import Model
from LanguageClass import Lang
import matplotlib.pyplot as plt
import os
class ServerClass():
    def __init__(self, params, options, train_data, val_data, eng_obj, alpha, data_list):
        self.params = params
        self.options = options
        self.train_data = train_data
        self.val_data = val_data
        self.eng_obj = eng_obj
        self.steps_for_validation = options['steps_for_validation']

        if self.options['algorithm']=='apfl':
            alphas = np.zeros((self.params['NUM_CLIENTS']))
            v = np.float(self.params['init_alpha'])

            for i in range(self.params['NUM_CLIENTS']):
                alphas[i] = v
        else:
            alphas = np.zeros(
                (self.params['NUM_CLIENTS'], self.params['NUM_CLIENTS']))
            v = np.float(self.params['init_alpha'])

            for i in range(self.params['NUM_CLIENTS']):
                for j in range(self.params['NUM_CLIENTS']):
                    alphas[i][j] = v
        models = []
        self.alphas = alphas
        self.data_list = data_list
        logger.info(self.options)
        # self.options['device']=torch.device("cpu")
        for i in range(self.params['NUM_CLIENTS']):
            temp_model = {}
            temp_model['global'] = Model(
                options, eng_obj.embeddings).to(self.options['device'])
            if self.options['algorithm']!='fedavg':
              temp_model['private'] = Model(
                  options, eng_obj.embeddings).to(self.options['device'])
              temp_model['personalized'] = Model(
                  options, eng_obj.embeddings).to(self.options['device'])
            if self.options['algorithm']=='papfl':
              temp_model['private_copy'] = Model(
                  options, eng_obj.embeddings).to(self.options['device'])
            models.append(temp_model)

        client_ids = [i for i in range(self.params['NUM_CLIENTS'])]
        self.models = models
        if self.options['algorithm']!='papfl':
            self.opts = [optim.Adam(self.models[i]['global'].parameters(), lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        self.client_ids = client_ids
        # self.opts=[optim.Adam(self.models[i]['global'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        if self.options['algorithm']!='fedavg':
          self.opts_priv = [optim.Adam(self.models[i]['private'].parameters(
          ), lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
    def val_func(self, net, val_sents):
        """
            Function for doing the validation when data_as_stream flag is  set
        """
        n_total = 0
        n_loss = 0
        # Can modify to do on a restricted set of the sentences
        # net
        # global val_factor
        val_factor = 0.1
        # to make the training faster if a large validation set is there then restricting the datalength to be used
        # val_factor is a config parameter (0 to 1)
        n = int(len(val_sents)*val_factor)
    #     print(n)
        # setting the model in eval mode
        net.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=self.eng_obj.pad_token_id)
        valloader = DataLoader(
            val_sents, batch_size=self.options['batch_size'], shuffle=True)
        with torch.no_grad():
            for batch_idx, (inp, fwd, bwd) in enumerate(itertools.islice(valloader, int(n/self.options['batch_size']))):
                inp = inp.to(self.options['device'])
                bwd = bwd.to(self.options['device'])
                fwd = fwd.to(self.options['device'])
    #             xlm_inp=xlm(inp)
                out = net(inp)
                bwd_sent = bwd.view(-1)
                fwd_sent = fwd.view(-1)
                targs = torch.cat([fwd_sent, bwd_sent], dim=0)
                loss = criterion(out, targs.long())
                # print(loss)
                n_loss += (loss.item())
            avg_loss = n_loss/(batch_idx+1)
        return avg_loss

    def federated_train(self,):
        cur_loss_global = []
        cur_loss_local = []
        gradlist = []
        n_items = 0
        for i in self.client_ids:
            self.models[i], loss_global, loss_local, self.alphas[i] = self.local_train(
                self.models[i], self.options['learning_rate'], self.train_data[i], self.alphas[i], i)
            cur_loss_local.append(sum(loss_local))
            cur_loss_global.append(sum(loss_global))
        return self.models, sum(cur_loss_global)/len(self.client_ids), sum(cur_loss_local)/len(self.client_ids)

    def plot_metrics(self):
      algorithm=self.options['algorithm']
      file_types=['global','local','val']
      #todo modify this to use master_path 
      file_name=self.options['folder_path']+os.sep+'algo_file_loss.hdf5'
      for files in file_types:
        plt.figure()
        file=file_name.replace("algo",algorithm).replace("file",files)
        logger.info(file)
        if not os.path.isfile(file):
          logger.info(file+" not found")
          continue
        fp=h5py.File(file,'r')
        acc=np.array(fp.get(files+'_loss')[:])
        
        plt.title(files)
        # logger.info(acc)
        if files=='val':
          acc=np.sum(acc,axis=1)/acc.shape[1]
          # logger.info(acc)
        plt.plot(acc)
        # plt.figure()
        plt.savefig(self.options['folder_path']+os.sep+"plot_"+algorithm+"_"+files+str(acc[-1])+".png")
    def save_results(self, loss_values, loss_values_local, val_loss):
        save_model=self.options['save_model']
        logger.info("Saved alphas")
        logger.info(self.alphas)
        if loss_values is not None and len(loss_values) != 0:
            logger.info("Saving global training loss")
            with h5py.File(self.options['folder_path']+os.sep+self.options['algorithm']+'_global_loss.hdf5', 'w') as fp:
                fp.create_dataset('global_loss', data=np.array(loss_values))
                fp.close()
        if loss_values_local is not None and len(loss_values_local) != 0:
            logger.info("Saving local training loss")
            with h5py.File(self.options['folder_path']+os.sep+self.options['algorithm']+'_local_loss.hdf5', 'w') as fp:
                fp.create_dataset(
                    'local_loss', data=np.array(loss_values_local))
                fp.close()
        if val_loss is not None and len(val_loss) != 0:
            logger.info("Saving validation loss")
            with h5py.File(self.options['folder_path']+os.sep+self.options['algorithm']+'_val_loss.hdf5', 'w') as fp:
                fp.create_dataset('val_loss', data=np.array(val_loss))
                fp.close()
        if save_model == 1:
            if self.options['algorithm']=='fedavg':
              model_path = self.options['folder_path']+os.sep+\
                    "_"+self.options['algorithm']+"_" + \
                    "_model.pt"
              logger.info("Saving Model  at"+model_path)
              torch.save(
                    self.models[i]['personalized'].state_dict(), model_path)
            else:
              for i in range(self.params['NUM_CLIENTS']):
                  model_path = self.options['folder_path']+os.sep+\
                      "_"+self.options['algorithm']+"_" + \
                      self.data_list[i][0]+"_model.pt"
                  logger.info("Saving Model  at"+model_path)
                  torch.save(
                      self.models[i]['personalized'].state_dict(), model_path)
        else:
            print("nont svaing model")
