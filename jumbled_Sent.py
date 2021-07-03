import sys
import numpy as np
import os
def swap(wlist,inds):
    temp=wlist[inds[0]]
    wlist[inds[0]]=wlist[inds[1]]
    wlist[inds[1]]=temp
    return wlist
def jumble_sent(data):
    jumbled_data=[]
    for i in range(len(data)):

            sentlist=[]
            for sent in data[i].split("\t"):
                wlist=[]
                for words in sent.split():
                    wlist.append(words)
                n=len(wlist)
                for i in range(2):
                    inds=np.random.randint(n,size=2)
                    wlist=swap(wlist,inds)
                s=" ".join(wlist)
                sentlist.append(s)
            jumbled_data.append(sentlist[0]+'\t'+sentlist[1])
        
           
    return jumbled_data
def main():
    filename=sys.argv[1]
    cwd=os.getcwd()
    print(filename)
    with open(filename,'r',encoding='utf-8') as fp:
        data=fp.readlines()
    jd=jumble_sent(data)
    jfile="jumbled_"+filename
    with open(jfile,"w",encoding='utf-8') as fp:
        for item in jd:
            fp.write(item+"\n")


if __name__=="__main__":
    main()
