from Language_modelling_training_colab import *
import json
import tensorflow as tf
import os
import tensorflow_hub as hub
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('INFO')
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#   except RuntimeError as e:
#     # Visible devices must be set at program startup
#     print(e)
embed = hub.load(r"https://tfhub.dev/google/universal-sentence-encoder/4")
def set_cpu_option():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# set_cpu_option()
def run_sts_benchmark(sent1,sent2):
    sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(sent1)), axis=1)
    sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(sent2)), axis=1)
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    """Returns the similarity scores"""
    return scores
options={}
options['max_seq_len']=40
dataset1=['euro']
dataset2=['supreme','movie','word_pred','Taskmaster','euro']
# dataset2=['euro']
print("_-------------------------_")
dict_mapping={}
for corp1 in dataset1:
    options['run_name']='euro'
    train_sents_1,_=get_sents(options,name=corp1)
    sent1=[" ".join(s for s in train_sents_1[:52000])]
    for corp2 in dataset2:
        
        if corp1==corp2:
            continue
        else:
            print(corp1," ",corp2)
            options['run_name']=corp2
            train_sents_2,_=get_sents(options,name=corp2)
#         print(train_sents_1[:10],train_sents_2[:10]) 
        sent2=[]
        sent2=[" ".join(s for s in train_sents_2[:52000])]
        temp_dict={}
        temp_dict[corp2]=run_sts_benchmark(sent1,sent2)
        if corp1 in dict_mapping:
            dict_mapping[corp1].append(temp_dict)    
        else:
            dict_mapping[corp1]=[temp_dict]
#         print(run_sts_benchmark(sent1,sent2))
print(dict_mapping)