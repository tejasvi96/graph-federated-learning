# one_time file
# from google.colab import drive
# drive.mount('/content/drive')

# !pip install pytorch-nlp
from torchnlp.word_to_vector import GloVe
import os
import pickle
import numpy as np
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

def one_time(master_path,vocab_file):
# for data should be a one time thing only

#     os.system("tar -xzvf "+master_path+"/movieqa.tar.gz -C '/content/drive/MyDrive'") 
#     os.system("tar -xzvf "+master_path+"/CBTest.tgz -C "+master_path )
#     os.system("unzip "+ master_path+ "/wikitext-2-raw-v1.zip -d "+master_path)
#     os.system("tar -xzvf "+ master_path+"/movieqa.tar.gz -C "+master_path)
#     os.system(" tar -xzvf "+master_path+"/swb1_dialogact_annot.tar.gz -C "+master_path)
#     # os.system(" tar -xzvf 'drive/MyDrive/ubuntu_dialogs.tgz' -C '/content/drive/MyDrive'")
# # !


# # For different runs 

#     os.mkdir(master_path+'/supreme')
#     os.mkdir(master_path+'/children')
#     os.mkdir(master_path+'/wiki')
#     os.mkdir(master_path+'/reddit')

    fname=master_path+"/"+vocab_file
    with open(fname,'r') as fp:
      data=fp.readlines()
    google_words=[]

    special_tokens=['<s>','</s>','<unk>','<pad>']
    # for tok in special_tokens:
      # google_words.append(tok)pi
    list_of_words=['.',',','?','!']
    google_words=google_words+list_of_words
    for word in data:
      google_words.append(word.split("\n")[0])
    pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in google_words)
    embedding_weights=np.zeros((len(google_words)+4,300))
    eng_master=Lang('Master')

    for ind,word in enumerate(special_tokens):
      embedding_weights[ind]=np.random.randn(300)
    for ind,word in enumerate(google_words):
      embedding_weights[ind+4]=pretrained_embedding[word] 
      eng_master.addWord(word)


    eng_master.embeddings=embedding_weights
    file_path=master_path+'/FL_model/embedding_class_corrected.pt'
    picklefile = open(file_path, 'wb')
    pickle.dump(eng_master, picklefile)
    picklefile.close()
    picklefile = open(file_path, 'rb')
    tempobj=pickle.load(picklefile)

def env_setup(master_path,vocab_file):
#     master_path=os.getcwd()
#     master_path='/content/drive/MyDrive/'
    # os.system('pip install -r '+master_path+'requirements.txt')
    os.system('pip install pytorch-nlp')
    os.system('apt install openjdk-8-jdk')
# nnAA
    # os.system('update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java')
    # os.system('pip install language-check')
    #os.system('pip install pycontractions')
    one_time(master_path,vocab_file)

# if __name__=="__main__":
#     master_path=os.getcwd()
# #     master_path='/content/drive/MyDrive/'
#     os.system('pip install -r '+master_path+'requirements.txt')
#     os.system('pip install pytorch-nlp')
#     os.system('apt install openjdk-8-jdk')
# # nnAA
#     os.system('update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java')
#     os.system('pip install language-check')
#     os.system('pip install pycontractions')
#     one_time(master_path)
#     # main()