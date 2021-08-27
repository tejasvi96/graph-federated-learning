import h5py
import numpy as np
import matplotlib.pyplot as plt
def plotting(type):
  # plt.figure();
  for file in hdf5files:
    fp=h5py.File(file,'r')
    ind=file.find("_")
    ind1=file.find("_",ind+1)
    ind2=file.find("_",ind1+1)
    ind3=file.find(".",ind2+1)
    algo=file[ind1+1:ind2]
    # if algorithm!=algo:
      # continue
    print(algo)
    run_type=file[ind2+1:ind3]
    print(run_type)
    if run_type!=type:
      continue
    acc=np.array(fp.get(run_type)[:])
    x=[i for i in range(acc.shape[0])]
    if 'val' in run_type:
      print(acc)
      acc=np.sum(acc,axis=1)/acc.shape[1]
      x=[i*10 for i in x]  
    # print(acc)
    # print(x)
    print(algo+"  :  "+str(acc[-1]))
    plt.plot(x,acc,label=run_type+"_"+algo)
    plt.legend()
    plt.title(run_type)
  plt.show()
  

import os
# cwd=os.getcwd()
files=os.listdir('./Results')
hdf5files=[os.getcwd()+os.sep+"Results"+os.sep+f for f in files if f.endswith(".hdf5")]
print(hdf5files)
plotting('val_loss')
# plotting("global_loss")
plotting("local_loss")