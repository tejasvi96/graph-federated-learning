# training
import numpy as np
import pandas as pd
import json
import pickle
# need a class having the embeddings as well
import torch
import math
from torch.utils.data import TensorDataset
# arr_sents=stream_load(ubuntu_cleaned_sents,eng,mapping_dict)
import torch
import pickle
from loguru import logger
from allennlp.modules.elmo_lstm import ElmoLstm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset,Subset
import torch.nn as nn
import itertools
def stream_load(eng,sents,options):
    """
        The function to be used for loading the entire data as a stream of text and not using  the padding token.
        Inputs:
        class lang object for word2index index2word mapping dictionaries.
        sents: The list of raw language input sentences
        returns a tensordataset
    """
    # max_seq_len=eng.max_seq_len
    tokenized_sents=[i for i in range(len(sents))]
    for i,sent in enumerate(sents):
        # For bidrectionalism making use of only </s> token 
        # sent=sent +" </s>"
        tokenized_sents[i]=[eng.word2index[word] if word in eng.word2index.keys() else eng.unk_token_id for word in sent.split(' ')]
        # This is the end token
        tokenized_sents[i].append(1)
    # The list of all the sentence tokens conatenated as a one list 
    merged_sents=list(itertools.chain.from_iterable(tokenized_sents))
    
    # As the vocabulary is restricted the inp_sents will have the token id of the tokenizer and fwd_sents and bwd_sents (targets) will have the token id of the restricted vocab
    inp_sents=[]
    fwd_sents=[]
    bwd_sents=[]
    n=len(merged_sents)

    # The main code to implement the bidrectionalism 
    # The data stream looks like this
    # Data Stream: This is a boy</s> He is a good king.</s> There is a cat ...
    # seq_len:11
    # inp: This is a boy </s> He is a good king</s>
    # fwd: is a boy </s> He is a good king </s> There
    # bwd: There This is a boy </s> He is a good king
    max_seq_len=options['max_seq_len']
    for i in range(0,n,max_seq_len):
        temp=merged_sents[i:i+max_seq_len]
        inp_sents.append(temp)
        temp=[]
        # Start the index from i+1 token 
        for j in range(i+1,i+max_seq_len+1 if n>i+max_seq_len+1 else n):
            temp.append(merged_sents[j])
        fwd_sents.append(temp)
        temp=[]
        # Append the last token first
        temp.append(merged_sents[i+max_seq_len-1 if n>i+max_seq_len else n-1])
        for j in range(i,i+max_seq_len-1 if n>i+max_seq_len else n-1):
            temp.append(merged_sents[j])
        bwd_sents.append(temp)


    # To finally store the list as an array the last sentence split may not have max_seq_len tokens thus using the padding token 
    if len(inp_sents[-1])<max_seq_len:
        inp_sents[-1]=inp_sents[-1]+[eng.pad_token_id]*(max_seq_len-len(inp_sents[-1]))
    if len(fwd_sents[-1])<max_seq_len:
        fwd_sents[-1]=fwd_sents[-1]+[eng.pad_token_id]*(max_seq_len-len(fwd_sents[-1]))
    if len(bwd_sents[-1])<max_seq_len:
        bwd_sents[-1]=bwd_sents[-1]+[eng.pad_token_id]*(max_seq_len-len(bwd_sents[-1]))
    
    # Converting the lists to an array to be used with TensorDataset
    arr_inp_sents=np.array(inp_sents,dtype=int)
    arr_fwd_sents=np.array(fwd_sents,dtype=int)
    arr_bwd_sents=np.array(bwd_sents,dtype=int)

    #TensorDataset returned to the preprocess function
    arr_sents=TensorDataset(torch.from_numpy(arr_inp_sents),torch.from_numpy(arr_fwd_sents),torch.from_numpy(arr_bwd_sents))
    return arr_sents


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pickle
# Training and validation split
from tqdm import tqdm
def count_parameters(model):
    """
        Helper function to print the count of trainable parameters 
        Inputs:
        Model: the model object
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model(model):
    """
        Helper function to print the model architecture
        Inputs:
        model: The model object
    """
    #Todo
    #currently Hardcoded XLM
    logger.info(xlm.parameters)
    logger.info(model.parameters)
def norm_calc(model):
    """
        The helper function to calculate the norm of the gradients of the model's parameters. To be used in conjunction with gradient clipping
        Inputs:
        model:The Model object
        returns the norm as a floating point value
    """
    total_norm=0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    logger.info("Gradient norm "+str(total_norm))

def train_func(arr_sents,val_sents,scheduler,opt,min_val_loss,options):

    """
        The training function loop to be used when data_as_stream flag is  set
        Inputs:
        scheduler: The Learning Rate scheduler object initialized using the configuration parameters
        opt: The optimizer object (Here we are using Adam)
        min_val_loss: This is set and used when we are reusing the model for training and we initiazie it to the loss using the loaded model in model_setup function
    """
    logger.info("Started Training on "+options['run_name'])

    #writer object for tensorboard
    writer = SummaryWriter()
    
    #starting the model training
    net.train()
    files_dir='./'
    run_name=options['run_name']
    match=options['match']
    if match is None:
        match=""
    #main training loop
    device=options['device']
    validation_loss_values=[]
    train_loss_values=[]
    for epc in range(options['epochs']):
        trainloader=DataLoader(arr_sents,batch_size=options['batch_size'],shuffle=True)
        n_totals=0
        n_loss=0
        # To check whether the weight decay has been done at least once in this epoch
        weight_decay_flag=0
        for batch_idx,(inp,fwd,bwd) in tqdm(enumerate(trainloader)):
#             if batch_idx==10:
#                 break
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
#             xlm_inp=xlm(inp)
            out=net(inp)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            opt.zero_grad()
#             print(out)
#             print(targs)
            loss=criterion(out,targs.long())
            loss.backward()
            print(loss.item())
            
            train_loss_values.append(loss.item())
            n_loss+=loss.item()
            n_totals+=inp.shape[0]
            opt.step()
            avg_loss=n_loss/(batch_idx+1)

            if (batch_idx+1) %options['steps_for_validation']==0:
                val_loss=val_func(val_sents,options)
                net.train()
                logger.info("After "+str(batch_idx+1)+" steps Training Avg_loss "+str(n_loss/(batch_idx+1))+"Training Avg_perplexity "+str(math.exp(n_loss/(batch_idx+1)))+" "+ "Validation Avg_loss "+str(val_loss)+"Validation Avg_perplexity "+str(math.exp(val_loss)))
                #setting weight decay for the larger datasets
                weight_decay_flag=1
                scheduler.step(val_loss)
                if val_loss<min_val_loss:
                    logger.info("Saved the model state best validation loss ")
                    min_val_loss=val_loss
                    model_path=files_dir+run_name+"/"+match+"_"+"model_best.pt"
                    optim_path=files_dir+run_name+"/"+match+"_"+"optim_best.pth"
#                     torch.save(opt.state_dict(),optim_path)
                    torch.save(net.state_dict(),model_path)
                validation_loss_values.append(val_loss)

        avg_loss=n_loss/(batch_idx+1)
        logger.info("Epoch "+str(epc+1))
        val_loss=val_func(val_sents,options)
        net.train()
        # norm_calc(net)

        # If the weight decay has not been done even once in the epoch then do weight decay (meant for small datasets)
        if weight_decay_flag==0:
            scheduler.step(val_loss)
            if val_loss<min_val_loss:
                logger.info("Saved the model state best validation loss ")
                min_val_loss=val_loss
                model_path=files_dir+run_name+"/"+match+"_"+"model_best.pt"
                optim_path=files_dir+run_name+"/"+match+"_"+"optim_best.pth"
#                 torch.save(opt.state_dict(),optim_path)
                torch.save(net.state_dict(),model_path)            
        
        logger.info("Training Avg_loss "+str(avg_loss)+"Training Avg_perplexity "+str(math.exp(avg_loss))+" "+ "Validation Avg_loss "+str(val_loss)+"Validation Avg_perplexity "+str(math.exp(val_loss)))
        writer.add_scalar("Perplexity/Val",math.exp(val_loss),epc+1)
        writer.add_scalar('Perplexity/Train',math.exp(avg_loss), epc+1)
        
        # Saving the state after 10 epochs 
        if (epc+1) %10==1:
            logger.info("Saved the model state after "+str(epc+1)+" epochs")
            model_path=files_dir+run_name+"/"+match+"_"+"model_"+str(epc+1)+".pt"
            optim_path=files_dir+run_name+"/"+match+"_"+"optim_"+str(epc+1)+".pth"
#             torch.save(opt.state_dict(),optim_path)
            torch.save(net.state_dict(),model_path)
    
        val_file='./'+run_name+"_val.txt"
        with open(val_file,'w') as fp:
            for lvalue in validation_loss_values:
                fp.write(str(lvalue)+"\n")
        return train_loss_values,val_loss_values

def val_func(val_sents,options):
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
    logger.info(str(n)+" sents used for validation")
    # setting the model in eval mode
    net.eval()
    device=options['device']
    valloader=DataLoader(val_sents,batch_size=options['batch_size'],shuffle=True)
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd) in enumerate(itertools.islice(valloader,int(n/options['batch_size']))):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
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



def model_setup(options,arr_sents,val_sents,eng):
    """
        Main function to load and define the model
        Does the job of creating the model object and loads exisiting model if load_model flag is set
        If do_pretrain is set then calls the appropriate train function
        If do_eval is set then calls the appropriate test function
    """

    global net,device,xlm
    data_as_stream=1
    load_model=options['load_model']
    load_model_file=options['load_model_file']
    do_training=1
    # check whether the current device has the cuda support
    is_cuda = torch.cuda.is_available()
#     print(is_cuda)
    # initialize device to cpu and override if the cuda is set and available 
#     device= torch.device("cpu")
    device=options['device']
    # is_cuda checks for availability of GPU on the user machine
    # cuda is a config parameter whether the user wants to use the GPU or not.
#     if is_cuda :
#         device = torch.device("cuda")
    # Todo 
    # Using the fixed xlm model only now(non trainable)
#     xlm=XLMModel.from_pretrained(pretrained_model_name)
#     logger.info("Successfully loaded the XLM model")
#     xlm=xlm.to(device)

    # using the data_as_stream option choosing which model to use
    if data_as_stream==0:
        net=Model_pad(options)
    else:
        net=Model(options,eng.embeddings)
    
    # Moving the model to device 
    print(device)
    net=net.to(device)
    net=net.double()
    # The optimizer object currently hardcoded to Adam
    opt=optim.Adam(net.parameters(),lr=options['learning_rate'])
    
    global criterion

    # Here the index should be tokenizer.pad_index 
    # For XLMTokenizer it is 2 and is not included in the Cross Entropy Loss calculation
    criterion=nn.CrossEntropyLoss(ignore_index=eng.pad_token_id)

    #initializing the minimum validation loss to a higher value
    min_val_loss=100

    #if the existing model is to be loaded then loading the model using this config params
    if load_model==1:
        # added map location so that model trained on gpu can be loaded on cpu too
        net.load_state_dict(torch.load(load_model_file,map_location=device))
        # Load optimizer only if training has to be done again not to be used if doing only eval 
        logger.info("Model successfully Loaded")
        if do_training==1:
#             opt.load_state_dict(torch.load(load_optim_file))
            if data_as_stream==0:
                min_val_loss=val_func_pad()
            else:
                min_val_loss=val_func(val_sents)
#             logger.info("Initial Validation loss "+str(min_val_loss))
            # Initialing the minimum validation loss also so as when trianing again the best takes the loaded model as a baseline
        
        
#     logger.info("Trainable Parameters")
#     logger.info(count_parameters(net))
#     print_model(net)

    # Weight_decay scheduler
    # set to min mode as loss is decreasing
    # Weight_decayfactor min_lr are the config parameters can be set there 
    scheduler = ReduceLROnPlateau(opt, mode='min',min_lr=options['min_learning_rate'],factor=options['weight_decay_factor'], patience=0, verbose=True,threshold=1e-4)
    if do_training==1:
        if data_as_stream==0:
            train_func_pad(scheduler,opt,min_val_loss,options)
        else:
            train_loss_values,val_loss_values=train_func(arr_sents,val_sents,scheduler,opt,min_val_loss,options)
    return train_loss_values,val_loss_values
#     if do_eval==1:
#         if data_as_stream==0:
#             test_func_pad()
#         else:
#             test_func()
# nn.CrossEntropyLoss?
# testing for the movie data
import torch
import pickle
from torch.utils.data import TensorDataset
import os
import numpy as np
# from pycontractions import Contractions
# cont = Contractions(api_key='glove-twitter-100')
# cont.load_models()# function to expand contractions
# def expand_contractions(text):
#     text = list(cont.expand_texts([text], precise=True))[0]
#     return text
import re# function to remove special characters
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()# call function
from torch.utils.data import TensorDataset
def get_ubuntu_data():
    fpath='/content/drive/MyDrive/dialogs'
    dirs=os.listdir(fpath)
    ubuntu_data=[]
    for folder in dirs:
        datadirs=os.listdir(fpath+"/"+folder)
        if len(datadirs)==0:
            continue
        for files in datadirs:
            temp_data=[]
            with open(fpath+"/"+folder+'/'+files,'r',encoding='utf-8') as fp:
                data=fp.readlines()
            for lines in data:
                line=lines.split("\t")[3]
                temp_data.append(line)
            ubuntu_data.append(temp_data)    
    temp=[" ".join(sent for sent in para) for para in ubuntu_data ]
    return temp

def get_supreme_data():
    fpath='./supreme_court_dialogs_corpus_v1.01/supreme_court_dialogs_corpus_v1.01'
    print(os.listdir(fpath))
    fname='supreme.conversations.txt'
    with open(fpath +"//"+fname, 'r') as  fp:
        data=fp.readlines()
    supreme_court_data=[]
    for line in data:
        ln=line.split('+++$+++')
        supreme_court_data.append(ln[7])  
    return supreme_court_data

def get_movie_data():
    fpath='./cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'
    fname='movie_lines.txt'
    with open(fpath+"/"+fname,'r',encoding='unicode_escape') as fp:
        data=fp.readlines()
    movie_dialog_data=[]
    for line in data:
        movie_dialog_data.append(line.split('+++$+++')[4])
    return movie_dialog_data

def get_word_prediction_data():
    # remember to remove <s> and </s> and unk and other tokens
    fpath='./WordPrediction/unked-clean-dict-15k'
    os.listdir(fpath)
    fname='en-sents-shuf.00.valid.txt'
    with open(fpath + "//"+ fname,'r') as fp:
        word_pred_data=fp.readlines()
    
    word_pred_data=[sent.replace('</s>'," ").replace("<s>"," ") for sent in word_pred_data]
    return word_pred_data

def get_reddit_data(match):
    cwd='/content/drive/MyDrive/'
    path= cwd+ r'reddit-dataset'
    fnames=(os.listdir(path))
    print(fnames)
    
    conv_files=[]

    for files in fnames:
    #     print(files)
        if match in files and files.endswith('.csv'):
            conv_files.append(files)
    data=pd.read_csv(path+"//"+conv_files[0])
    data=list(data['0'])
    data=[re.sub(r'(\s)www\w+', r'\1',sent) for sent in data if sent is not np.nan]
    return data
def get_taskmaster_data():
    master_path='./Taskmaster'
    dirs=['TM-1-2019', 'TM-2-2020/data/']
    conv=[]
    for folder in dirs:
        files=[file for file in os.listdir(master_path+"/"+folder) if file.endswith('.json')]
        filepath=master_path+"/"+folder
        for file in files:
            if 'ontology' in file or 'sample' in file:
                continue
#             print(filepath+"/"+file)
            with open(filepath+"/"+file,'r') as fp:
                data=json.load(fp)
#             print(data)
            for i in range(len(data)):
                for line,sent in enumerate(data[i]['utterances']):
                    conv.append(sent['text'])
    return conv
def get_eurodata():
    filename='./euro_parls_train_eng.txt'
    with open(filename,'r',encoding='utf-8') as fp:
        data=fp.readlines()
    return data
def get_sents(options,name):
    match=""
    if name=='ubuntu':
        temp=get_ubuntu_data()
    elif name == 'supreme':
        temp=get_supreme_data()
    elif name=='movie':
        temp=get_movie_data()
    elif name=='word_pred':
        temp=get_word_prediction_data()
    elif name=='reddit':
        temp=get_reddit_data(match)
    elif name=='Taskmaster':
        temp=get_taskmaster_data()
    elif name=='euro':
        temp=get_eurodata()
    else:
        logger.info("Not found ",name)
        return "",""
#     return temp
    normalized_sents=[normalizeString(sent) for sent in temp]
    # arr_sents=stream_load(normalized_sents,ubuntu_eng)
    logger.info('name:'+name)
    logger.info("Sentences:"+str(len(normalized_sents)))

    ubuntu_cleaned=[remove_extra_whitespace_tabs(sent) for sent in normalized_sents]
#     ubuntu_cleaned2=[expand_contractions(sent) for sent in ubuntu_cleaned]
    # return ubuntu_cleaned,ubuntu_cleaned2
    
    # values=np.array(list(eng.word2count.values()))
    # value=np.quantile(values,0.90)
    
    # for k,v in eng.word2count.items():
    #     if v<value:
    #         del(eng.word2index[k])
    # eng.n_words=len(eng.word2index)
    file_path='./embedding_class_corrected.pt'
    picklefile = open(file_path, 'rb')
    master_obj=pickle.load(picklefile)
    embedding_weights=master_obj.embeddings
    # pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in eng.word2index.keys())
    # embedding_weights = np.zeros((eng.n_words, pretrained_embedding.dim))
    # embedding_weights=np.zeros((eng.n_words,300))    
    # # will help in restricting vocab by having less sizes of embedding matrices
    # mapping_dict={}

    # # will help in decoding
    # mapping_dict_invert={}
    # for ind, token in enumerate(eng.word2index.keys()):
    #     mapping_dict[eng.word2index[token]]=ind
    #     mapping_dict_invert[ind]=eng.word2index[token]
    #     embedding_weights[ind] = pretrained_embedding[token]
    ubuntu_cleaned_sents=[" ".join(w for w in sent.split(" ")) for sent in ubuntu_cleaned]
    file_dir='./'+name
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_path=file_dir+"/"+'vocab_'+match+'.pt'
    if not os.path.exists(file_path):
      eng=Lang(name+"_"+match)
      for sent in ubuntu_cleaned:
        for word in sent.split(" "):
            eng.addWord(word)
      picklefile = open(file_path, 'wb')
      pickle.dump(eng, picklefile)
      picklefile.close()
    else:
      picklefile = open(file_path, 'rb')
      eng = pickle.load(picklefile)
    logger.info("Vocabulary size:"+str(eng.n_words))

    #todo
    # threshold parameter can be used here to cap the number of sentences
    logger.info("Some sentences:")
    logger.info(ubuntu_cleaned_sents[:5])
    split_factor=0.95
    split_index=int(split_factor*len(ubuntu_cleaned_sents))
    train_sents=ubuntu_cleaned_sents[:split_index]
    val_sents=ubuntu_cleaned_sents[split_index:]
    return train_sents,val_sents

def swap(wlist,inds):
    temp=wlist[inds[0]]
    wlist[inds[0]]=wlist[inds[1]]
    wlist[inds[1]]=temp
    return wlist
def jumble_sent(data,jumbles):
    jumbled_data=[]
    for i in range(len(data)):
            sentlist=[]
            for sent in data[i].split("\t"):
#                 print(sent)
                wlist=[]
                for words in sent.split():
                    wlist.append(words)
                n=len(wlist)
#                 print(wlist)
                for i in range(min(jumbles,n)):
                    inds=np.random.randint(n,size=jumbles)
                    wlist=swap(wlist,inds)
                s=" ".join(wlist)
                sentlist.append(s)
            jumbled_data.append(sentlist[0])
    return jumbled_data

def data_preprocess(options,name,match=None,jumbled=0):

    if match is None:
        match=""
    if jumbled==1:
      train_loaded_file=options['data_dir']+os.sep+name+os.sep+'train_data_'+match+'.pt'
      val_loaded_file=options['data_dir']+os.sep+name+os.sep+'val_data_'+match+'.pt'
    else:
      train_loaded_file=options['data_dir']+os.sep+name+os.sep+'train_data_'+match+options['jumbled']+'.pt'
      val_loaded_file=options['data_dir']+os.sep+name+os.sep+'val_data_'+match+options['jumbled']+'.pt'    
    logger.info("Checking existence of "+train_loaded_file)

    if os.path.exists(train_loaded_file):
        train_sents=torch.load(train_loaded_file)
        logger.info("loaded "+train_loaded_file)
        if os.path.exists(val_loaded_file):
            val_sents=torch.load(val_loaded_file)
        else:
            logger.info('val_file not found')
            val_sents=None
        return train_sents,val_sents
    if name=='ubuntu':
        temp=get_ubuntu_data()
    elif name == 'supreme':
        temp=get_supreme_data()
    elif name=='movie':
        temp=get_movie_data()
    elif name=='word_pred':
        temp=get_word_prediction_data()
    elif name=='reddit':
        temp=get_reddit_data(match)
    elif name=='Taskmaster':
        temp=get_taskmaster_data()
    elif name=='euro':
        temp=get_eurodata()
    else:
        logger.info("New dataset not found",name)
#     return temp
    normalized_sents=[normalizeString(sent) for sent in temp]
    # arr_sents=stream_load(normalized_sents,ubuntu_eng)
    logger.info('name:'+name)
    logger.info("Sentences:"+str(len(normalized_sents)))

    ubuntu_cleaned=[remove_extra_whitespace_tabs(sent) for sent in normalized_sents]
#     ubuntu_cleaned2=[expand_contractions(sent) for sent in ubuntu_cleaned]
    # return ubuntu_cleaned,ubuntu_cleaned2
    
    # values=np.array(list(eng.word2count.values()))
    # value=np.quantile(values,0.90)
    
    # for k,v in eng.word2count.items():
    #     if v<value:
    #         del(eng.word2index[k])
    # eng.n_words=len(eng.word2index)
    file_path=options['master_path']+'/embedding_class_corrected.pt'
    picklefile = open(file_path, 'rb')
    master_obj=pickle.load(picklefile)
    embedding_weights=master_obj.embeddings
    # pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in eng.word2index.keys())
    # embedding_weights = np.zeros((eng.n_words, pretrained_embedding.dim))
    # embedding_weights=np.zeros((eng.n_words,300))    
    # # will help in restricting vocab by having less sizes of embedding matrices
    # mapping_dict={}

    # # will help in decoding
    # mapping_dict_invert={}
    # for ind, token in enumerate(eng.word2index.keys()):
    #     mapping_dict[eng.word2index[token]]=ind
    #     mapping_dict_invert[ind]=eng.word2index[token]
    #     embedding_weights[ind] = pretrained_embedding[token]
    ubuntu_cleaned_sents=[" ".join(w for w in sent.split(" ")) for sent in ubuntu_cleaned]
    file_dir=options['master_path']+name
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_path=file_dir+"/"+'vocab_'+match+'.pt'
    if not os.path.exists(file_path):
      eng=Lang(name+"_"+match)
      for sent in ubuntu_cleaned:
        for word in sent.split(" "):
            eng.addWord(word)
      picklefile = open(file_path, 'wb')
      pickle.dump(eng, picklefile)
      picklefile.close()
    else:
      picklefile = open(file_path, 'rb')
      eng = pickle.load(picklefile)
    logger.info("Vocabulary size:"+str(eng.n_words))

    #todo
    # threshold parameter can be used here to cap the number of sentences
    # print(ubuntu_cleaned_sents[:10])
    if options['jumbled']!='':
        logger.info("jumbled")
       
#         was 4 earlier
        ubuntu_cleaned_sents=jumble_sent(ubuntu_cleaned_sents,jumbles=8)
        # print(ubuntu_cleaned_sents[:10])
    split_factor=0.95
    split_index=int(split_factor*len(ubuntu_cleaned_sents))
    train_sents=ubuntu_cleaned_sents[:split_index]
    val_sents=ubuntu_cleaned_sents[split_index:]
    train_sents=stream_load(master_obj,train_sents,options)
    val_sents=stream_load(master_obj,val_sents,options)
    logger.info("Instances after stream loading:"+str(len(train_sents)))
    #todo
    # add the logic to pickle the train_sents and val_sents and add check to stop the preprocessing if already present the pickled file
    # follow a naming convention
    torch.save(train_sents,train_loaded_file)
    torch.save(val_sents,val_loaded_file)
    return train_sents,val_sents
# data_list=[('supreme',None),('movie',None),('word_pred',None),('reddit','askscience'),('reddit','news_politics')]

# def get_data_lists(options):
#     data=[]
#     val_data=[]
#     for ind in range(len(data_list)):
#       #Comment this for non jumbled      
#       train_sents,val_sents=data_preprocess(options,name=data_list[ind][0],match=data_list[ind][1],jumbled=options['jumbled_run_flags'][ind])
#       data.append(train_sents)
#       val_data.append(val_sents)
#     return data,val_data

def train_independently(options,train_data,val_data,eng_obj):
    train_loss=[]
    val_loss=[]
    for ind in range(0,len(data_list),1):
        options['run_name']=data_list[ind][0]
        options['match']=data_list[ind][1]
        # logobj=logger.add(options['master_path']+options['run_name']+".txt")
        logger.info(options)
        train_sents,val_sents=train_data[ind],val_data[ind]
        num_train_samples=52000
        sample_ds = Subset(train_sents, np.arange(num_train_samples))
        train,val=model_setup(options,sample_ds,val_sents,eng_obj)
        logger.remove(logobj)   
        train_loss.append(train)
        val_loss.append(val)
    return train_loss,val_loss

# if __name__=="__main__":
#     options={}
#     options['batch_size']=128
#     options['max_seq_len']=40
#     options['num_lstm_layers']=1
#     options['hid_size']=300
#     options['dropout']=0.1
#     options['learning_rate']=0.01
#     options['min_learning_rate']=0.0001
#     options['weight_decay_factor']=0.5
#     options['run_name']='supreme'
#     options['epochs']=2
#     options['steps_for_validation']=10
#     options['run_type']=0
#     options['device']='cuda:2'
#     options['match']=None
# #     options['jumbled']='jumbled'
    
#     file_path='./embedding_class_corrected.pt'
#     picklefile = open(file_path, 'rb')
#     temp=pickle.load(picklefile)
#     options['vocab_size']=temp.n_words
#     train_data,val_data=get_data_lists(options)
#     options['load_model']=0
#     options['load_model_file']=None
#     files_dir='./'
#     for ind in range(0,5,1):
#         options['run_name']=data_list[ind][0]
#         options['match']=data_list[ind][1]
#         logobj=logger.add('./'+options['run_name']+".txt")
#         logger.info(options)
#         train_sents,val_sents=train_data[ind],val_data[ind]
#         num_train_samples=52000
#         sample_ds = Subset(train_sents, np.arange(num_train_samples))
#         model_setup(options,sample_ds,val_sents,temp)
#         logger.remove(logobj)