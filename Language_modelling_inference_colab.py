#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getopt
# import 
import copy
import torch
device=torch.device('cpu')
import os
import pickle
import numpy as np
import torch.nn as nn
import itertools
import numpy as np
import sys
import pickle
options={}
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
from torch.utils.data import TensorDataset
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
import torch.quantization
import re# function to remove special characters
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()# call function
from torch.utils.data import TensorDataset
class BiLMEncoder(ElmoLstm):
    """Wrapper around BiLM to give it an interface 
       Basically an lstm cell with a projection (reduced size than standard lstm if multiple layers are used)
    """

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * 2
class Model(nn.Module):
    """ 
        The LM Model to be used in case of the Streamed Input
        Inherits from the base class nn.Module
        init and forward are defined
    """
    def __init__(self,options,embedding_weights):
        """
            Options is a dictionary initialized using the configuration parameters like 
            hid_size: The hidden size returned by the input module (Here we are using XLM-17-1280 so  it has a value of 1280)
            dropout: The dropout value to be used in the BiLM and optionally in the linear layer
            num_lstm_layers: The number of stacked layers of bilm
            vocab_size: To initialize the final fully connected layer size
        """
        super(Model, self).__init__()
#         logger.info(options)
        self.embedding=nn.Embedding(options['vocab_size'],300).from_pretrained(torch.tensor(embedding_weights,dtype=torch.double))
        self.bilm=BiLMEncoder(300,options['hid_size'],options['hid_size'],options['num_lstm_layers'],recurrent_dropout_probability=options['dropout'])
        self.lin=torch.nn.Linear(options['hid_size'],options['vocab_size'])
        self.dropout=nn.Dropout(p=options['dropout'])
    def forward(self,enc_embedding):
        """
            This takes as input the output from the xlm module
            of shape (batch_size,max_seq_len,hid_size)
            Returns a tensor of shape (2*hid_size*batch_size)*vocab_size
            (2 is due to the bidirectionalism of bilm)
        """
        # This is ideally a mask of ones and used as it is with streamed input
        # Mask of ones of shape batch_size*max_seq_len
#         print(enc_embedding[0].shape)
        inp=enc_embedding.to(torch.int64)
#         inp=inp.to(device)
#         print(inp.shape)
#         print(inp)
        enc_embedding=self.embedding(inp)
#         print(enc_embedding.dtype)
        enc_embedding=enc_embedding.float()
        # print(enc_embedding.shape)
        mask=torch.ones((enc_embedding.shape[0],options['max_seq_len'])).to(device)
#         print(device)
        enc=self.bilm(enc_embedding,mask) 
        # returns tensor of size 1*batch_size*max_seq_len*(2*hid_size)
        # Here 2 is due to bidirectionalism
        fwd,bwd=enc[:,:,:,:options['hid_size']],enc[:,:,:,options['hid_size']:]
        # fwd and bwd of size 1*batch_size*max_seq_len*hid_size
        logits_fwd=self.lin(fwd).view(enc_embedding.shape[0] * options['max_seq_len'], -1)
        logits_bwd=self.lin(bwd).view(enc_embedding.shape[0] * options['max_seq_len'], -1)
        # logits fwd and logits bwd each of sizes (batch_size*max_seq_len)*vocab_size
        logits=torch.cat((logits_fwd,logits_bwd),dim=0)
        # logits of sizes (2*batch_size*max_seq_len)*vocab_size
        return logits
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<s>", 1: "</s>",2:'<unk>',3:'<pad>'}
        self.n_words = 4  # Count SOS and EOS
        
        self.pad_token_id=3
        self.unk_token_id=2
        self.embeddings=None
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
import unicodedata,re
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def get_alpha(model1_path,model2_path,temp,options):
  model1=Model(options,temp.embeddings)
  device=torch.device('cpu')
  #movie_model
  model1.load_state_dict(torch.load(model1_path,map_location=torch.device('cpu')))
  model1=model1.double()
  #supreme model
  model2=Model(options,temp.embeddings)
  model2.load_state_dict(torch.load(model2_path,map_location=torch.device('cpu')))
  model2=model2.double()
  keys=list(model1.state_dict().keys())
  alphaval=0.5
  learning_rate=0.0001
  val=0.0
  for k in keys:
      if k=='embedding.weight':
          continue
      print(k)
      u=model1.state_dict()[k]
      v=model2.state_dict()[k]
      grad=model2.state_dict()[k]
  #     print((u-v).view(-1,1).shape)
      print(val)
      try:
          val+=( (torch.dot((u-v).reshape(-1,1).squeeze(dim=1),grad.reshape(-1,1).squeeze(dim=1) )).item()/(torch.norm(u-v).item()*torch.norm(grad).item()))
      except:
          pass
  alphaval=alphaval-learning_rate*val
  print(val)
  print(alphaval)

def generate_seq(net, eng, max_seq_len, seed_text, n_words,topk):
  import pandas as pd
  import copy
  df=pd.DataFrame({'sentence':[],'topk':[]})
#   temp={'sentence':"hi",'topk':'there'}
# df=df.append(temp,ignore_index=True)
  result = []
  in_text = seed_text
#   print(in_text)
#   options={}
  options['batch_size']=1
  options['max_seq_len']=max_seq_len
  # max_seq_len=3
  # generate a fixed number of words
  net.eval()
  with torch.no_grad():
    for _ in range(n_words):
      # encode the text as integer
      encoded = stream_load(eng,in_text,options)
      # truncate sequences to a fixed length
      # predict probabilities for each word
      # print(encoded[0][0])
      # print(encoded[0][1])
      yhat =net(torch.unsqueeze(encoded[0][0],dim=0))
#       print(yhat[options['max_seq_len']-1].shape)
      values,indices=(torch.topk(yhat[options['max_seq_len']-1],topk))
      # print(yhat[:options['max_seq_len'],:])
      
      ind=torch.argmax(yhat[:options['max_seq_len'],:],dim=1)

      out_word=eng.index2word[ind[-1].item()]
      out_words=[eng.index2word[index.item()] for index in indices] 
#       print(values)
#       print("top k suggestions")
      
#       print(out_words)
      preds=""
      for idx,word in enumerate(out_words):
        preds=preds +" "+word+" : "+str(round(values[idx].item(),2) )+","
#             print(word+" : "+str(round(values[idx].item(),2) ),end=',')
      # append to input
      temp={'sentence':copy.deepcopy(in_text),'topk':preds}
      df=df.append(temp,ignore_index=True)  
      if out_words[0]=='<unk>':
          in_text[0]+=' '+out_words[1]
          result.append(out_words[1])
      else:      
          in_text[0]+=' '+out_words[0]
          result.append(out_words[0])
#       print(in_text)
      
      options['max_seq_len']+=1
#   print(df['sentence'].to_markdown(index=False))
  print(df)
  #print(df.to_markdown(index=False))
  return ' '.join(result)

def generate_seq_multiple(net, eng, max_seq_len, seed_text, n_words,topk,algo_list):
  import pandas as pd
  import copy
  df=pd.DataFrame({'fedavg':[],'apfl':[],'independent':[]})
#   temp={'sentence':"hi",'topk':'there'}
# df=df.append(temp,ignore_index=True)
  result = []
  in_text = seed_text
#   print(in_text)
#   options={}
  options['batch_size']=1
  options['max_seq_len']=max_seq_len
  # max_seq_len=3
  # generate a fixed number of words
  net.eval()
  with torch.no_grad():
    for _ in range(n_words):
      # encode the text as integer
      encoded = stream_load(eng,in_text,options)
      # truncate sequences to a fixed length
      # predict probabilities for each word
      # print(encoded[0][0])
      # print(encoded[0][1])
      yhat =net(torch.unsqueeze(encoded[0][0],dim=0))
#       print(yhat[options['max_seq_len']-1].shape)
      values,indices=(torch.topk(yhat[options['max_seq_len']-1],topk))
      # print(yhat[:options['max_seq_len'],:])
      
      ind=torch.argmax(yhat[:options['max_seq_len'],:],dim=1)

      out_word=eng.index2word[ind[-1].item()]
      out_words=[eng.index2word[index.item()] for index in indices] 
#       print(values)
#       print("top k suggestions")
      
#       print(out_words)
      preds=""
      for idx,word in enumerate(out_words):
        preds=preds +" "+word+" : "+str(round(values[idx].item(),2) )+","
#             print(word+" : "+str(round(values[idx].item(),2) ),end=',')
      # append to input
      temp={'sentence':copy.deepcopy(in_text),'topk':preds}
      df=df.append(temp,ignore_index=True)  
      if out_words[0]=='<unk>':
          in_text[0]+=' '+out_words[1]
          result.append(out_words[1])
      else:      
          in_text[0]+=' '+out_words[0]
          result.append(out_words[0])
#       print(in_text)
      
      options['max_seq_len']+=1
#   print(df['sentence'].to_markdown(index=False))
  print(df)
  #print(df.to_markdown(index=False))
  return ' '.join(result)

def stream_load(eng,sents,options):
    """
        The function to be used for loading the entire data as a stream of text and not using  the padding token.
        Inputs:
        sents: The list of raw language input sentences
        tokenizer: The tokenizer to be used for tokenizing the words of sentence (Here makes use of the XLM Tokenizer)
        restricted_vocab: The dictionary  which maps the xlm word token indices to restricted vocab indices
        max_seq_len: The configuration parameter max_seq_len
        returns a tensordataset
    """
    # max_seq_len=eng.max_seq_len
    tokenized_sents=[i for i in range(len(sents))]
    for i,sent in enumerate(sents):
        # For bidrectionalism making use of only </s> token 
        # sent=sent
        # sent=sent +" </s>"
        tokenized_sents[i]=[eng.word2index[word] if word in eng.word2index.keys() else eng.unk_token_id for word in sent.split(' ')]
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

def get_loss(sent,net,options,temp):
    """ sent is a string """
    """temp is the lang object"""
#     sent="you know how sometimes you just become this persona and you don t know how to quit ."
    # sent='Hi , my name is john .'
    normalized_sents=[normalizeString(sent)]
        # arr_sents=stream_load(normalized_sents,ubuntu_eng)
    ubuntu_cleaned=[remove_extra_whitespace_tabs(sent) for sent in normalized_sents]
    #ubuntu_cleaned2=[expand_contractions(sent) for sent in ubuntu_cleaned]
    ubuntu_cleaned_sents=[" ".join(w for w in sent.split(" ")) for sent in ubuntu_cleaned]
    print(ubuntu_cleaned_sents)

    t=stream_load(temp,ubuntu_cleaned_sents,options)
    print(t[0])
    net.eval()
    with torch.no_grad():
      out=net(torch.unsqueeze(t[0][0],dim=0))
      bwd_sent=t[0][2].view(-1)
      fwd_sent=t[0][1].view(-1)
      targs=torch.cat([fwd_sent,bwd_sent],dim=0)
      #print(targs.shape)
      #print(out.shape)
    #print(temp.pad_token_id)
    criterion=nn.CrossEntropyLoss(ignore_index=temp.pad_token_id)
    print(criterion(out,targs))



    
def main(argv):
    print(argv)
    try:
      opts, args = getopt.getopt(argv,"s:l:p:t:a:q:k:j:",["string=","length=","predwords=","typ=",'algo=','quantized=','topk=','path='])
    except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
#     print(opts)
    master_path=os.getcwd()
    for opt, arg in opts:
        print(opt,arg)
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-s", "--string"):
            sent = [arg]
        elif opt in ("-l", "--length"):
            cur_sent_length=int(arg)
        elif opt in ("-p", "--predwords"):
            generate_words=int(arg)
        elif opt in ("-t","--typ"):
            model_dir=arg.split()
        elif opt in ("-a","--algo"):
            algo=arg
        elif opt in ("-q","--quantized"):
            quantisation=bool(int(arg))
        elif opt in ("-k","--topk"):
            topk=int(arg)
        elif opt in ("-j","--path"):
            master_path=arg
#             model_dir=list(" ".join(word for word in arg.split()))
#     print(model_dir)    
#     model_dir='movie'
    print(master_path)
    options['batch_size']=32
    options['max_seq_len']=40
    options['num_lstm_layers']=1
    options['hid_size']=300
    options['dropout']=0.1
    options['learning_rate']=0.001
    options['min_learning_rate']=0.0001
    options['weight_decay_factor']=0.8
    options['run_name']='ubuntu'
    options['epochs']=1
    options['steps_for_validation']=500
    
    file_path=master_path+'\\embedding_class_corrected.pt'
    picklefile = open(file_path, 'rb')
    temp=pickle.load(picklefile)
    options['vocab_size']=temp.n_words

#     get_loss(sent[0],net,options,temp)
    algo_list=['apfl','independent','fedavg']
    gen_sequence=1
    print("Algorithm Used for Training", algo.upper())
    if gen_sequence==1:
        sents=sent
        # sent='Hi , my name is john .'
        normalized_sents=[normalizeString(sent) for sent in sents]
            # arr_sents=stream_load(normalized_sents,ubuntu_eng)
        ubuntu_cleaned=[remove_extra_whitespace_tabs(sent) for sent in normalized_sents]
        # ubuntu_cleaned2=[expand_contractions(sent) for sent in ubuntu_cleaned]
        ubuntu_cleaned_sents=[" ".join(w for w in sent.split(" ")) for sent in ubuntu_cleaned]
        
        for mode in model_dir:
            if algo=='independent':
                model_file=master_path+mode+'\\_model_best.pt'
            elif algo=='gapfl':
                model_file=master_path+'\\apfl_'+mode+'_model.pt'
            else:
                model_file=master_path+'\\fedavg_model.pt'
            net=Model(options,temp.embeddings)

#             if torch.cuda.is_available:
#                 device=torch.device("cuda:3")
#             else:
            device=torch.device("cpu")
            net.load_state_dict(torch.load(model_file,map_location=device))
#             net=net.double()
            if quantisation:
                net = torch.quantization.quantize_dynamic(net, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
#             print_size_of_model(net)
            copied=copy.deepcopy(ubuntu_cleaned_sents)
            print("Model:"+mode)
            _=generate_seq(net,temp,cur_sent_length,copied,generate_words,topk)

if __name__=="__main__":
    main(sys.argv[1:])