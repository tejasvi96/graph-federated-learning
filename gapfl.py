from Language_modelling_training_colab import *
import numpy as np
from tqdm import tqdm
# options={}
# options['batch_size']=128
# options['max_seq_len']=40
# options['num_lstm_layers']=1
# options['hid_size']=300
# options['dropout']=0.1
# options['learning_rate']=0.01
# options['min_learning_rate']=0.0001
# options['weight_decay_factor']=0.5
# options['run_name']='supreme'
# options['epochs']=1
# options['steps_for_validation']=500
# options['match']=None
# options['device']=torch.device('cuda:0')
# file_path='./embedding_class_corrected.pt'
# params={}
# params['NUM_CLIENTS']=2
# params['learning_rate']=0.005
# params['init_alpha']=None
# params['num_rounds']=1502
# params['tau']=5
# picklefile = open(file_path, 'rb')
# temp=pickle.load(picklefile)
# options['vocab_size']=temp.n_words
# options['run_type']=0
# options['jumbled']='jumbled'
# # data_list=[('supreme',None),('euro',None),('reddit','askscience'),('reddit','news_politics')]
# data_list=[('supreme',None),('movie',None),('word_pred',None),('Taskmaster',None),('euro',None)]
# data_list=[('supreme',None),('movie',None),('word_pred',None)]
# data_list=[('supreme',None),('movie',None),('word_pred',None),('euro',None)]
# data_list=[('word_pred',None),('Taskmaster',None)]
# data_list=[('supreme',None),('supreme',None),('euro',None),('Taskmaster',None),('Taskmaster',None)]
# data_list=[('supreme',None),('supreme',None),('supreme',None),('supreme',None),('supreme',None)]
# data_list=[('word_pred',None),('movie',None),('supreme',None),('Taskmaster',None),('euro',None),('supreme',None),('movie',None),('word_pred',None),('Taskmaster',None),('euro',None)]
# def get_data_lists(options):
#     data=[]
#     val_data=[]
#     for ind in range(len(data_list)):
#         if ind==8:
#             options['jumbled']='jumbled_8'
#         else:
#             del options['jumbled']
#         if ind<3 and options['run_type']:
#             options['run_name']=data_list[0][0]
#             train_sents,val_sents=data_preprocess(options,name=data_list[0][0],match=data_list[0][1])
#         elif options['run_type']:
#             options['run_name']=data_list[1][0]
#             train_sents,val_sents=data_preprocess(options,name=data_list[1][0],match=data_list[1][1])
#         else:
#             options['run_name']=data_list[ind][0]
#             train_sents,val_sents=data_preprocess(options,name=data_list[ind][0],match=data_list[ind][1])
#         data.append(train_sents)
#         val_data.append(val_sents)
#     return data,val_data
import torch.optim as optim
import numpy as np

class gapfl():
    def __init__(self,params,options,train_data,val_data,eng_obj,alpha,data_list):
        self.params=params
        self.options=options
        self.train_data=train_data
        self.val_data=val_data
        self.eng_obj=eng_obj
#         self.alphas=alphas
    
        alphas=np.zeros((self.params['NUM_CLIENTS'],self.params['NUM_CLIENTS']))
        v=np.float(self.params['init_alpha'])

        for i in range(self.params['NUM_CLIENTS']):
            for j in range(self.params['NUM_CLIENTS']):
                if i!=j:
                    alphas[i][j]=v
        models=[]
        self.alphas=alphas
        self.data_list=data_list
        for i in range(self.params['NUM_CLIENTS']):
            temp_model={}
            temp_model['global']=Model(options,eng_obj.embeddings).to(self.options['device'])
            temp_model['private']=Model(options,eng_obj.embeddings).to(self.options['device'])
            temp_model['personalized']=Model(options,eng_obj.embeddings).to(self.options['device'])
                                             
            models.append(temp_model)
        
        client_ids=[i for i in range(self.params['NUM_CLIENTS'])]
        self.models=models
        self.opts=[optim.Adam(self.models[i]['global'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        self.client_ids=client_ids
        self.opts=[optim.Adam(self.models[i]['global'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        self.opts_priv=[optim.Adam(self.models[i]['private'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
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
        def batch_fn(model, batch,flag,ind):
            return self.batch_train(model, batch, learning_rate,flag,ind)
        l_local=[]
        l_global=[]
        flag=0
        k=0
        all_batches=DataLoader(train_sents,batch_size=self.options['batch_size'],shuffle=True)
        for idx,batch in enumerate(all_batches):
            if idx==1:
                break
            flag=1
            model['global'],losses,_=batch_fn(model,batch,0,ind)
            l_global.append(losses)
            model['private'],losses,grad_dict=batch_fn(model,batch,1,ind)
            l_local.append(losses)
        return model,l_global,l_local,alphaval

    def federated_train(self,):
        cur_loss_global=[];
        cur_loss_local=[]
        gradlist=[]
        n_items=0
        for i in self.client_ids:
            self.models[i],loss_global,loss_local,self.alphas[i]=self.local_train(self.models[i],self.options['learning_rate'],self.train_data[i],self.alphas[i],i)
            cur_loss_local.append(sum(loss_local))
            cur_loss_global.append(sum(loss_global))
        return self.models,sum(cur_loss_global)/len(self.client_ids),sum(cur_loss_local)/len(self.client_ids)


    def val_func(self,net,val_sents):
        """
            Function for doing the validation when data_as_stream flag is  set
        """
        n_total=0
        n_loss=0
        # Can modify to do on a restricted set of the sentences
        # net
        # global val_factor
        val_factor=0.1
        # to make the training faster if a large validation set is there then restricting the datalength to be used
        # val_factor is a config parameter (0 to 1)
        n=int(len(val_sents)*val_factor)
    #     print(n)
        # setting the model in eval mode
        net.eval()
        criterion=nn.CrossEntropyLoss(ignore_index=self.eng_obj.pad_token_id)
        valloader=DataLoader(val_sents,batch_size=self.options['batch_size'],shuffle=True)
        with torch.no_grad():
            for batch_idx,(inp,fwd,bwd) in enumerate(itertools.islice(valloader,int(n/self.options['batch_size']))):
                inp=inp.to(self.options['device'])
                bwd=bwd.to(self.options['device'])
                fwd=fwd.to(self.options['device'])
    #             xlm_inp=xlm(inp)
                out=net(inp)
                bwd_sent=bwd.view(-1)
                fwd_sent=fwd.view(-1)
                targs=torch.cat([fwd_sent,bwd_sent],dim=0)
                loss=criterion(out,targs.long())
                # print(loss)
                n_loss+=(loss.item())
            avg_loss=n_loss/(batch_idx+1)
        return avg_loss

    def save_results(self,personalized_accs):
    #     file_name='./results_cs_bad_correl_jumbled'+str(options['run_type'])+"_NLP"+"_fixedalpha_"+str(params['learning_rate'])+"_.txt"
        file_name=self.options['master_path']+'results_'+self.options['expt']+"val.txt"
        with open(file_name,'a') as fp: 
            for item in lossValues:
                fp.write(str(item.item())+'\n')
            for item in lossValueslocal:
                fp.write(str(item.item())+'\n')
    #         fp.write(str(params)+"\n")
    #         fp.write(str(options)+"\n")
    #         fp.write(str(alphas)+"\n")
    #         fp.write(" ".join(str(ac) for ac in priv_accs)+"\n")
    #         fp.write(" ".join(str(gc) for gc in global_accs)+"\n")
            for acc in personalized_accs:
                fp.write(" ".join(str(pc) for pc in acc)+"\n")
            fp.write(str(data_list))

    def model_train(self):
        priv_accs=[]
        personalized_accs=[]
        global_accs=[]
        temp_acc=[]
        update_alpha=1
        lossValues=[]
        lossValueslocal=[]
        degree=4.0
        for round_num in range(self.params['num_rounds']):
            temp_acc=[]
            print("-------------------------")
            print("Round:",str(round_num))
            print("-------------------------")
            if((round_num+1)%self.params['tau']==0):
                print(self.alphas)

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
                        for c2 in range(self.params['NUM_CLIENTS']):
                            if c==c2:
                                continue
                            v=0.0
                            norm1=0.0
                            norm2=0.0
                            for param in self.models[c]['private'].named_parameters():
                                if param[1].grad is None:
                                    continue
                                grads=param[1].grad
                                diff=self.models[c]['private'].state_dict()[param[0]][:]-self.models[c2]['global'].state_dict()[param[0]][:]
                                norm1+=torch.norm(diff,p='fro').item()
                                diff=diff.reshape(-1,1)
                                norm2+=torch.norm(grads,p='fro').item()
                                grads=grads.reshape(-1,1)
                                val=torch.dot(grads.squeeze(dim=1),diff.squeeze(dim=1)).float()
                                v+=val
                            v=v/(norm1*norm2)
                            self.alphas[c][c2]=self.alphas[c][c2] - self.options['alpha_lr']*v.item()*4.0
                            if self.alphas[c][c2]>1:
                                self.alphas[c][c2]=1.0
                            if self.alphas[c][c2]<0:
                                self.alphas[c][c2]=0.0

                    for param in self.models[c]['personalized'].named_parameters():
                        self.models[c]['personalized'].state_dict()[param[0]][:]=torch.zeros_like(self.models[c]['personalized'].state_dict()[param[0]])
                        for c2 in range(self.params['NUM_CLIENTS']):
                            if c==c2:
                                continue
                            self.models[c]['personalized'].state_dict()[param[0]][:]+=(self.models[c]['private'].state_dict()[param[0]][:]*(self.alphas[c][c2]/degree)+self.models[c2]['global'].state_dict()[param[0]][:]*((1.0-self.alphas[c][c2])/degree))
        #             print(models[c]['personalized'].state_dict()['embedding.weight'])
            lossValues.append(prev_loss_global)
            lossValueslocal.append(prev_loss_local)
            if round_num%10==0:
                for ind in range(self.params['NUM_CLIENTS']):
#                 priv_accs.append(val_func(models[ind]['private'],val_data[ind]))
#                 global_accs.append(val_func(models[ind]['global'],val_data[ind]))
                    temp_acc.append(self.val_func(self.models[ind]['personalized'],self.val_data[ind]))                    
                personalized_accs.append(temp_acc)
        for ind in range(self.params['NUM_CLIENTS']):
#                 priv_accs.append(val_func(models[ind]['private'],val_data[ind]))
#                 global_accs.append(val_func(models[ind]['global'],val_data[ind]))
            temp_acc.append(self.val_func(self.models[ind]['personalized'],self.val_data[ind]))            
        personalized_accs.append(temp_acc)
        self.save_results(personalized_accs)
        if save_model==1:
            for i in range(self.params['NUM_CLIENTS']):
                model_path=self.options['master_path']+"apfl_"+self.data_list[i][1]+"_model.pt"
                torch.save(temp_model[i]['personalized'].state_dict(),model_path)

# if __name__=='__main__':
#     options['expt']='normal_run'
#     params['NUM_CLIENTS']=5
#     save_model=1
#     print(params)
#     print(options)
#     list_init_alphas=[0.3]
#     list_learning_rates=[1]
#     list_learning_rates=[1]
#     list_run_types=[0]
#     data,val_data=get_data_lists(options)
#     logger.info(data_list)
#     update_alpha=1
#     update_alphas=[1]
# #     logger.info(data_list)
# #     good alphas
# #     alphas=[[0,0.393,0.412,0.436,0.361],[0.393,0,0.444,0.3663,0.395],[0.412,0.444,0,0.45804,0.408],[0.436,0.3663,0.458,0,0.4138],[0.361,0.395,0.408,0.4138,0]]
# #     alphas=[[0,0.01,0.99,0.01,0.99],[0.01,0,0.01,0.99,0.01],[0.99,0.01,0,0.01,0.99],[0.01,0.99,0.01,0,0.01],[0.99,0.01,0.99,0.01,0]]
# #     alphas=[[0,0.0001,0.0001,0.0001,0.0001],[0.0001,0,0.50,0.50,0.50],[0.0001,0.50,0,0.50,0.50],[0.0001,0.50,0.50,0,0.50],[0.0001,0.50,0.50,0.50,0]]
# #     alphas=np.random.rand(5,5)
#     for upd_alpha in update_alphas:
#         logger.info('Update alpha  '+str(upd_alpha))
#         update_alpha=upd_alpha
#         for lrs in list_learning_rates:
#             params['learning_rate']=lrs
#             params['init_alpha']=0.0001
#             models=[]
#             for i in range(params['NUM_CLIENTS']):
#                 temp_model={}
#                 temp_model['global']=Model(options,temp.embeddings).to(options['device'])
#                 temp_model['private']=Model(options,temp.embeddings).to(options['device'])
#                 temp_model['personalized']=Model(options,temp.embeddings).to(options['device'])
#                 models.append(temp_model)
#             client_ids=[i for i in range(params['NUM_CLIENTS'])]
#             opts=[optim.Adam(models[i]['global'].parameters(),lr=options['learning_rate']) for i in range(params['NUM_CLIENTS'])]
#             opts_priv=[optim.Adam(models[i]['private'].parameters(),lr=options['learning_rate']) for i in range(params['NUM_CLIENTS'])]
#             prev_loss=0
#             lossValues=[]
#             lossValueslocal=[]
#             alphas=np.zeros((params['NUM_CLIENTS'],params['NUM_CLIENTS']))
#             v=np.float(params['init_alpha'])
        
#             for i in range(params['NUM_CLIENTS']):
#                 for j in range(params['NUM_CLIENTS']):
#                     if i!=j:
#                         alphas[i][j]=v

#             logger.info(str(alphas))
#             priv_accs=[]
#             personalized_accs=[]
#             global_accs=[]
#             temp_acc=[]
#             for round_num in range(params['num_rounds']):
#                 temp_acc=[]
#                 print("-------------------------")
#                 print("Round:",str(round_num))
#                 print("-------------------------")
#                 if((round_num+1)%params['tau']==0):
#                     print(alphas)

#                     for i in tqdm(range(len(client_ids))):
#                         if i==0:
#                             global_model=models[i]['global']
#                         else:
#                             for param in models[i]['global'].named_parameters():
#             #                     print(param[0])
#                                 global_model.state_dict()[param[0]][:]+=( models[i]['global'].state_dict()[param[0]][:])
#                     for param in models[0]['global'].named_parameters():
#                         global_model.state_dict()[param[0]][:]/=float(params['NUM_CLIENTS'])
#                     for i in range(len(client_ids)):
#                         for param in  models[i]['global'].named_parameters():
#                             models[i]['global'].state_dict()[param[0]][:]=global_model.state_dict()[param[0]][:]
#                 else:
#                     models,prev_loss_global,prev_loss_local=federated_train(models, data,client_ids)
#                     for c in range(params['NUM_CLIENTS']):
#                         if update_alpha:
#                             for c2 in range(params['NUM_CLIENTS']):
#                                 if c==c2:
#                                     continue
#                                 v=0.0
#                                 norm1=0.0
#                                 norm2=0.0
#                                 for param in models[c]['private'].named_parameters():
#                                     if param[1].grad is None:
#                                         continue
#                                     grads=param[1].grad
#                                     diff=models[c]['private'].state_dict()[param[0]][:]-models[c2]['global'].state_dict()[param[0]][:]
#                                     norm1+=torch.norm(diff,p='fro').item()
#                                     diff=diff.reshape(-1,1)
#                                     norm2+=torch.norm(grads,p='fro').item()
#                                     grads=grads.reshape(-1,1)
#                                     val=torch.dot(grads.squeeze(dim=1),diff.squeeze(dim=1)).float()
#                                     v+=val
#                                 v=v/(norm1*norm2)
#                                 alphas[c][c2]=alphas[c][c2] - params['learning_rate']*v.item()*4.0
#                                 if alphas[c][c2]>1:
#                                     alphas[c][c2]=1.0
#                                 if alphas[c][c2]<0:
#                                     alphas[c][c2]=0.0

#                         for param in models[c]['personalized'].named_parameters():
#                             models[c]['personalized'].state_dict()[param[0]][:]=torch.zeros_like(models[c]['personalized'].state_dict()[param[0]])
#                             for c2 in range(params['NUM_CLIENTS']):
#                                 if c==c2:
#                                     continue
#                                 models[c]['personalized'].state_dict()[param[0]][:]+=(models[c]['private'].state_dict()[param[0]][:]*(alphas[c][c2]/4.0)+models[c2]['global'].state_dict()[param[0]][:]*((1.0-alphas[c][c2])/4.0))
#             #             print(models[c]['personalized'].state_dict()['embedding.weight'])
#                 lossValues.append(prev_loss_global)
#                 lossValueslocal.append(prev_loss_local)
#                 if round_num%10==0:
#                     for ind in range(params['NUM_CLIENTS']):
# #                 priv_accs.append(val_func(models[ind]['private'],val_data[ind]))
# #                 global_accs.append(val_func(models[ind]['global'],val_data[ind]))
#                         temp_acc.append(val_func(models[ind]['personalized'],val_data[ind]))                    
#                     personalized_accs.append(temp_acc)
#             for ind in range(params['NUM_CLIENTS']):
# #                 priv_accs.append(val_func(models[ind]['private'],val_data[ind]))
# #                 global_accs.append(val_func(models[ind]['global'],val_data[ind]))
#                 temp_acc.append(val_func(models[ind]['personalized'],val_data[ind]))            
#             personalized_accs.append(temp_acc)
#             save_results(personalized_accs)
#             if save_model==1:
#                 for i in range(params['NUM_CLIENTS']):
#                     model_path="./apfl_"+data_list[i][1]+"_model.pt"
#                     torch.save(temp_model[i]['personalized'].state_dict(),model_path)
                    