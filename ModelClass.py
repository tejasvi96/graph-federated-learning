import torch
import torch.nn as nn
from allennlp.modules.elmo_lstm import ElmoLstm

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
        self.options=options
        
#         logger.info(options)
        self.embedding=nn.Embedding(options['vocab_size'],300).from_pretrained(torch.tensor(embedding_weights,dtype=torch.float))
        self.embedding.requires_grad=False
        self.bilm=BiLMEncoder(300,options['hid_size'],options['hid_size'],options['num_lstm_layers'],recurrent_dropout_probability=options['dropout'])
        self.lin=torch.nn.Linear(options['hid_size'],options['vocab_size'])
        self.dropout=nn.Dropout(p=options['dropout'])
    def forward(self,enc_embedding):
        """
            This takes as input the tokenized stream
            of shape (batch_size,max_seq_len,hid_size)
            Returns a tensor of shape (2*hid_size*batch_size)*vocab_size
            (2 is due to the bidirectionalism of bilm)
        """
        
        inp=enc_embedding.to(torch.int64)
        enc_embedding=self.embedding(inp)
        # This is ideally a mask of ones and used as it is with streamed input
        # Mask of ones of shape batch_size*max_seq_len
        mask=torch.ones((enc_embedding.shape[0],self.options['max_seq_len'])).to(self.options['device'])
        enc=self.bilm(enc_embedding,mask) 
        # returns tensor of size 1*batch_size*max_seq_len*(2*hid_size)
        # Here 2 is due to bidirectionalism
        fwd,bwd=enc[:,:,:,:self.options['hid_size']],enc[:,:,:,self.options['hid_size']:]
        # fwd and bwd of size 1*batch_size*max_seq_len*hid_size
        logits_fwd=self.lin(fwd).view(enc_embedding.shape[0] * self.options['max_seq_len'], -1)
        logits_bwd=self.lin(bwd).view(enc_embedding.shape[0] * self.options['max_seq_len'], -1)
        # logits fwd and logits bwd each of sizes (batch_size*max_seq_len)*vocab_size
        logits=torch.cat((logits_fwd,logits_bwd),dim=0)
        # logits of sizes (2*batch_size*max_seq_len)*vocab_size
        return logits
