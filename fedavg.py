# from Language_modelling_training_colab import *
# params={}
# params['NUM_CLIENTS']=5
# params['learning_rate']=0.1
# params['init_alpha']=0.5
# params['num_rounds']=1502
# params['tau']=10
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
# options['device']=torch.device('cuda:3')
# opt_priv'
# opt_priv'
import torch.optim as optim
import numpy as np
# Init all to same weights initially
from Language_modelling_training_colab import *
from tqdm import tqdm
class fedavg():
    def __init__(self,params,options,train_data,val_data,eng_obj):
        self.params=params
        self.options=options
        self.train_data=train_data
        self.val_data=val_data
        self.eng_obj=eng_obj

        
        models=[]

        for i in range(self.params['NUM_CLIENTS']):
            temp_model={}
            temp_model['global']=Model(options,eng_obj.embeddings).to(self.options['device'])
#             temp_model['private']=Model(options,temp.embeddings).to(options['device'])
#             temp_model['personalized']=Model(options,temp.embeddings).to(options['device'])
            models.append(temp_model)
        
        client_ids=[i for i in range(self.params['NUM_CLIENTS'])]
        self.models=models
        self.opts=[optim.Adam(self.models[i]['global'].parameters(),lr=self.options['learning_rate']) for i in range(self.params['NUM_CLIENTS'])]
        self.client_ids=client_ids

    def batch_train(self,initial_model,batch,learning_rate,flag,ind,options,eng):
        criterion=nn.CrossEntropyLoss(ignore_index=eng.pad_token_id)
        batch[0]=batch[0].to(options['device'])
        batch[1]=batch[1].to(options['device'])
        batch[2]=batch[2].to(options['device'])
        if flag==0:
            out=initial_model['global'](batch[0])
            bwd_sent=batch[2].view(-1)
            fwd_sent=batch[1].view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs.long())
            self.opts[ind].zero_grad()
            loss.backward()
    #         print(loss.item())

            self.opts[ind].step()
            return initial_model['global'],loss,loss
        else:
            out=initial_model['personalized'](batch[0])
            bwd_sent=batch[2].view(-1)
            fwd_sent=batch[1].view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs.long())

            print(loss.item())
    #         print(learning_rate)
            opt_per=optim.SGD(initial_model['private'].parameters(),lr=options['learning_rate'])
            opt_per.zero_grad()
    #         opt_priv[ind].zero_grad()
            loss.backward()
            grad_dict={}
            for param1,param2 in zip(initial_model['personalized'].named_parameters(),initial_model['private'].named_parameters()):
                if param1[1].grad is None:
                    continue
                grad_dict[param1[0]]=param1[1].grad.clone().detach()
                param2[1].grad=param1[1].grad.clone().detach()
            self.opt_per.step()
    #         opt_priv[ind].step()
            return initial_model['private'],loss,grad_dict
    def local_train(self,model, learning_rate, train_sents,ind,options,eng):
        def batch_fn(model, batch,flag,ind,options,eng):
            return self.batch_train(model, batch, learning_rate,flag,ind,options,eng)
        l_local=[]
        l_global=[]
        flag=0
        k=0
        all_batches=DataLoader(train_sents,batch_size=options['batch_size'],shuffle=True)
        for idx,batch in enumerate(all_batches):
            if idx==1:
                break
            flag=1
            model['global'],losses,_=batch_fn(model,batch,0,ind,options,eng)
            l_global.append(losses)
        return model,l_global
    def federated_train(self,data):
        cur_loss_global=[];
        cur_loss_local=[]
        gradlist=[]
        n_items=0
        for i in self.client_ids:
    #         print(alphas[i])
            self.models[i],loss_global=self.local_train(self.models[i],self.options['learning_rate'],data[i],i,self.options,self.eng_obj)
    #         print(alphas[i])
    #         cur_loss_local.append(sum(loss_local))
            cur_loss_global.append(sum(loss_global))

        return self.models,sum(cur_loss_global)/len(self.client_ids)

    def model_train(self):
    #     loss_500=[]
#         options=self.options
#         models=self.models
        val_loss=[]
        loss_temp=[]
        lossValues=[]
#         val_loss=[]
        for round_num in range(self.params['num_rounds']):
            loss_temp=[]
            print("-------------------------")
            print("Round:",str(round_num))
            print("-------------------------")
            if((round_num+1)%self.params['tau']==0):
                for i in tqdm(range(len(self.client_ids))):
                    if i==0:
                        global_model=self.models[i]['global']
                    else:
                        for param in self.models[i]['global'].named_parameters():
        #                     print(param[0])
                            global_model.state_dict()[param[0]][:]+=( self.models[i]['global'].state_dict()[param[0]][:])
                for param in self.models[i]['global'].named_parameters():
        #             if param[0]=='embedding.weight':
        #                 print(global_model.state_dict()[param[0]])
                    global_model.state_dict()[param[0]][:]/=float(self.params['NUM_CLIENTS'])
        #             if param[0]=='embedding.weight':
        #                 print(global_model.state_dict()[param[0]])
                for i in range(len(self.client_ids)):
                    for param in  self.models[i]['global'].named_parameters():
                        self.models[i]['global'].state_dict()[param[0]][:]=global_model.state_dict()[param[0]][:]
            else:
                self.models,prev_loss_global=self.federated_train(self.train_data)
            lossValues.append(prev_loss_global)

            if round_num%10==0:
                for ind in range(self.params['NUM_CLIENTS']):
                    loss_temp.append(self.val_func(self.models[ind]['global'],self.val_data[ind]))
                val_loss.append(loss_temp)
        for ind in range(self.params['NUM_CLIENTS']):
            loss_temp.append(self.val_func(self.models[ind]['global'],self.val_data[ind]))
        val_loss.append(loss_temp)
        return lossValues,val_loss

    def plot_results(self,lossValues):
        file_name='./results_temp.txt'
        with open(file_name,'w') as fp: 
            for item in lossValues:
                fp.write(str(item.item())+'\n')
            fp.write(str(params)+"\n")
            fp.write(str(options))
        lossv=[]
        with open(file_name,'r') as fp:
            data=fp.readlines()
        for val in data[:-2]:
            lossv.append(float(val.split("\n")[0]))
        import matplotlib.pyplot as plt
        plt.title("Plot of Training Loss vs iterations [FedAvg]")
        plt.xlabel('Iterations')
        plt.ylabel("Language Modelling Loss")
        plt.plot(lossv)
        plt.show()
    def save_results(self,lossValues,val_loss):
        file_name=self.options['master_path']+'results_feadavg'+str(self.options['run_type'])+"_"+str(self.options['learning_rate'])+'val.txt'
        with open(file_name,'a') as fp: 
    #         for item in lossValues:
    #             fp.write(str(item.item())+'\n')
    #         for item in lossValueslocal:
    #             fp.write(str(item.item())+'\n')
    #         fp.write(str(params)+"\n")
    #         fp.write(str(options)+"\n")
    #         fp.write(str(alphas)+"\n")
            for losses in val_loss:
                fp.write(" ".join(str(ac) for ac in losses)+"\n")
    #         fp.write(" ".join(str(gc) for gc in loss_1000)+"\n")
    #         fp.write(" ".join(str(pc) for pc in personalized_accs)+"\n")

    # data_list=[('supreme',None),('movie',None),('word_pred',None),('reddit','askscience'),('reddit','news_politics')]
    # data_list=[('supreme',None),('movie',None),('word_pred',None),('Taskmaster',None),('euro',None)]
#     def get_data_lists(options):
#         data=[]
#         val_data=[]
#         for ind in range(len(data_list)):
#     #         if ind<3 and options['run_type']:
#     #             options['run_name']=data_list[0][0]
#     #             train_sents,val_sents=data_preprocess(options,name=data_list[0][0],match=data_list[0][1])
#     #         elif options['run_type']:
#     #             options['run_name']=data_list[1][0]
#     #             train_sents,val_sents=data_preprocess(options,name=data_list[1][0],match=data_list[1][1])
#     #         else:
#     #             options['run_name']=data_list[ind][0]
#             train_sents,val_sents=data_preprocess(options,name=data_list[ind][0],match=data_list[ind][1])
#             data.append(train_sents)
#             val_data.append(val_sents)
#         return data,val_data
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
# if __name__=='__main__':
#     global opt_priv
#     save_model=1
#     options['run_type']=0
#     file_path='./embedding_class_corrected.pt'
#     picklefile = open(file_path, 'rb')
#     temp=pickle.load(picklefile)
#     options['vocab_size']=temp.n_words
# #     data_list=[('supreme',None),('movie',None),('word_pred',None),('reddit','askscience'),('reddit','news_politics')]
#     train_data=[]
#     val_data=[]
#     train_data,val_data=get_data_lists(options)
#     learning_rates=[0.01]
#     val_loss=[]
#     for lr in learning_rates:

#         options['learning_rate']=lr
#         models=[]
#         for i in range(params['NUM_CLIENTS']):
#             temp_model={}
#             temp_model['global']=Model(options,temp.embeddings).to(options['device'])
#     #         temp_model['private']=Model(options,temp.embeddings).to(options['device'])
#     #         temp_model['personalized']=Model(options,temp.embeddings).to(options['device'])
#             models.append(temp_model)

#         client_ids=[i for i in range(params['NUM_CLIENTS'])]
#         opts=[optim.Adam(models[i]['global'].parameters(),lr=options['learning_rate']) for i in range(params['NUM_CLIENTS'])]
#     #     opt_priv=[optim.Adam(models[i]['private'].parameters(),lr=options['learning_rate']) for i in range(params['NUM_CLIENTS'])]
#         prev_loss=0
#         lossValues=[]
#         lossValueslocal=[]
#         lossValues,val_loss=model_train(options,train_data,models,temp,opts)
#         print(val_loss)
# #         print(loss_1000)
#         save_results(lossValues,val_loss)
#     if save_model==1:
#         s="_".join(item[0] for item in data_list)
#         model_path="./"+s+"_model.pt"
#         torch.save(models[0]['global'].state_dict(),model_path)
        
#     plot_results(lossValues)