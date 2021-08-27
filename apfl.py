from ModelClass import Model
from ServerClass import ServerClass
from Language_modelling_training_colab import *
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import numpy as np
import h5py

class apfl(ServerClass):
    def __init__(self,params,options,train_data,val_data,eng_obj,alpha,data_list):
      ServerClass.__init__(self,params, options,
                                          train_data, val_data, eng_obj, alpha, data_list)

        # self.params=params
        # self.options=options
        # self.train_data=train_data
        # self.val_data=val_data
        # self.eng_obj=eng_obj
        # self.steps_for_validation=options['steps_for_validation']

    
        # alphas=np.zeros((self.params['NUM_CLIENTS']))
        # v=np.float(self.params['init_alpha'])

        # for i in range(self.params['NUM_CLIENTS']):
        #     alphas[i]=v
        # models=[]
        # self.alphas=alphas
        # self.data_list=data_list
        # logger.info(self.options)
        # # self.options['device']=torch.device("cpu")
        # for i in range(self.params['NUM_CLIENTS']):
        #     temp_model={}
        #     temp_model['global']=Model(options,eng_obj.embeddings).to(self.options['device'])
        #     temp_model['private']=Model(options,eng_obj.embeddings).to(self.options['device'])
        #     temp_model['personalized']=Model(options,eng_obj.embeddings).to(self.options['device'])
        #     models.append(temp_model)
        
        # client_ids=[i for i in range(self.params['NUM_CLIENTS'])]
        # self.models=models
        # self.opts=[optim.Adam(self.models[i]['global'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        # self.client_ids=client_ids
        # # self.opts=[optim.Adam(self.models[i]['global'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        # self.opts_priv=[optim.Adam(self.models[i]['private'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
    def batch_train(self,initial_model,batch,learning_rate,flag,ind):
        criterion=nn.CrossEntropyLoss(ignore_index=self.eng_obj.pad_token_id)
        
        batch[0]=batch[0].to(self.options['device'])
        batch[1]=batch[1].to(self.options['device'])
        batch[2]=batch[2].to(self.options['device'])
        if flag==0:
            out=initial_model['global'](batch[0])
            bwd_sent=batch[2].view(-1)
            fwd_sent=batch[1].view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs.long())
            self.opts[ind].zero_grad()
            loss.backward()
            self.opts[ind].step()
            return initial_model['global'],loss,loss
        else:
            out=initial_model['personalized'](batch[0])
            bwd_sent=batch[2].view(-1)
            fwd_sent=batch[1].view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs.long())
            self.opts_priv[ind].zero_grad()
            loss.backward()
            grad_dict={}
            for param1,param2 in zip(initial_model['personalized'].named_parameters(),initial_model['private'].named_parameters()):
                if param1[1].grad is None:
                    continue
                param2[1].grad=param1[1].grad.clone()
                param1[1].grad=None
            self.opts_priv[ind].step()
            return initial_model['private'],loss,grad_dict

    def local_train(self,model, learning_rate, train_sents,alph,ind):
        alphaval=alph
        # print(self.options['device'])
        def batch_fn(model, batch,flag,ind):
            return self.batch_train(model, batch, learning_rate,flag,ind)
        l_local=[]
        l_global=[]
        all_batches=DataLoader(train_sents,batch_size=self.options['batch_size'],shuffle=True)
        for idx,batch in enumerate(all_batches):
            if idx==1:
                break
            model['global'],losses,_=batch_fn(model,batch,0,ind)
            l_global.append(losses)
            model['private'],losses,grad_dict=batch_fn(model,batch,1,ind)
            l_local.append(losses)
        return model,l_global,l_local,alphaval


 
    #     file_name='./results_cs_bad_correl_jumbled'+str(options['run_type'])+"_NLP"+"_fixedalpha_"+str(params['learning_rate'])+"_.txt"
        """file_name=self.options['master_path']+'results_'+self.options['algorithm']+"val.txt"
        with open(file_name,'a') as fp: 
            for item in lossValues:
                fp.write(str(item)+'\n')
            for item in lossValueslocal:
                fp.write(str(item)+'\n')
    #         fp.write(str(params)+"\n")
    #         fp.write(str(options)+"\n")
    #         fp.write(str(alphas)+"\n")
    #         fp.write(" ".join(str(ac) for ac in priv_accs)+"\n")
    #         fp.write(" ".join(str(gc) for gc in global_accs)+"\n")
            for acc in personalized_accs:
                fp.write(" ".join(str(pc) for pc in acc)+"\n")
            fp.write(str(data_list))"""

    def model_train(self):
        val_loss=[]
        temp_acc=[]
        update_alpha=self.options['adaptive_alpha']
        lossValues=[]
        lossValueslocal=[]
        for round_num in range(self.params['num_rounds']):
            temp_acc=[]
            # logger.info("-------------------------")
            
            # logger.info("-------------------------")
            if((round_num+1)%self.params['tau']==0):
                logger.info("Round:"+str(round_num))
                logger.info(self.alphas)

                for i in tqdm(range(len(self.client_ids))):
                    if i==0:
                        global_model=self.models[i]['global']
                    else:
                        for param in self.models[i]['global'].named_parameters():
        #                     print(param[0])
                            global_model.state_dict()[param[0]][:]+=( self.models[i]['global'].state_dict()[param[0]][:])
                for param in self.models[0]['global'].named_parameters():
                    global_model.state_dict()[param[0]][:]/=float(self.params['NUM_CLIENTS'])
                for i in range(len(self.client_ids)):
                    for param in  self.models[i]['global'].named_parameters():
                        self.models[i]['global'].state_dict()[param[0]][:]=global_model.state_dict()[param[0]][:]
            else:
                self.models,prev_loss_global,prev_loss_local=self.federated_train()
                for c in range(self.params['NUM_CLIENTS']):
                    if update_alpha:
                      v=0.0
                      norm1=0.0
                      norm2=0.0
                      for param in self.models[c]['private'].named_parameters():
                          if param[1].grad is None:
                              continue
                          grads=param[1].grad
                          # diff=self.models[c]['private'].state_dict()[param[0]][:]-self.models[c2]['global'].state_dict()[param[0]][:]
                          diff=self.models[c]['private'].state_dict()[param[0]][:]-self.models[c]['global'].state_dict()[param[0]][:]
                      
                          norm1+=torch.norm(diff,p='fro').item()
                          diff=diff.reshape(-1,1)
                          norm2+=torch.norm(grads,p='fro').item()
                          grads=grads.reshape(-1,1)
                          val=torch.dot(grads.squeeze(dim=1),diff.squeeze(dim=1)).float()
                          v+=val
                      v=v/(norm1*norm2)
                      self.alphas[c]=self.alphas[c] - self.options['alpha_lr']*v.item()
                      if self.alphas[c]>1:
                          self.alphas[c]=1.0
                      if self.alphas[c]<0:
                          self.alphas[c]=0.0

                      for param in self.models[c]['personalized'].named_parameters():
                          self.models[c]['personalized'].state_dict()[param[0]][:]=torch.zeros_like(self.models[c]['personalized'].state_dict()[param[0]])
                          # for c2 in range(self.params['NUM_CLIENTS']):
                              # if c==c2:
                                  # continue
                              # self.models[c]['personalized'].state_dict()[param[0]][:]+=(self.models[c]['private'].state_dict()[param[0]][:]*(self.alphas[c][c2]/degree)+self.models[c2]['global'].state_dict()[param[0]][:]*((1.0-self.alphas[c][c2])/degree))
                          self.models[c]['personalized'].state_dict()[param[0]][:]=(self.models[c]['private'].state_dict()[param[0]][:]*(self.alphas[c])+self.models[c]['global'].state_dict()[param[0]][:]*((1.0-self.alphas[c])))
                    # else:
                      # print("Not updating Alpha")        
        #             print(models[c]['personalized'].state_dict()['embedding.weight'])
            lossValues.append(prev_loss_global.item())
            lossValueslocal.append(prev_loss_local.item())
            if round_num%self.steps_for_validation==0:
                for ind in range(self.params['NUM_CLIENTS']):
#                 priv_accs.append(val_func(models[ind]['private'],val_data[ind]))
#                 global_accs.append(val_func(models[ind]['global'],val_data[ind]))
                    temp_acc.append(self.val_func(self.models[ind]['personalized'],self.val_data[ind]))                    
                val_loss.append(temp_acc)
        for ind in range(self.params['NUM_CLIENTS']):
#                 priv_accs.append(val_func(models[ind]['private'],val_data[ind]))
#                 global_accs.append(val_func(models[ind]['global'],val_data[ind]))
            temp_acc.append(self.val_func(self.models[ind]['personalized'],self.val_data[ind]))            
        val_loss.append(temp_acc)
        self.save_results(lossValues,lossValueslocal,val_loss)
        return lossValueslocal, val_loss
