from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score
data=[]
from sklearn import metrics
import time
suffix=""#objects/

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-flag', action='store', default=None, dest='flag')
parser.add_argument('-nu', action='store', default=None, dest='nu')
parser.add_argument('-gama', action='store', default=None, dest='gama')
parser.add_argument('-gpu', action='store', default=None, dest='gpu')
args = parser.parse_args()
flag=int(args.flag)
nu_=float(args.nu)
gama_=float(args.gama)

path=['training_preprocessed_logs_S1-CVE-2015-5122_windows','training_preprocessed_logs_S2-CVE-2015-3105_windows','testing_preprocessed_logs_S3-CVE-2017-11882_windows','training_preprocessed_logs_S4-CVE-2017-0199_windows_py']
f=open("./embedding_data/"+suffix+"S"+str(flag)+"_benign.json")
number_list=['S1_number_.npy','S2_number_.npy','S3_number_.npy','S4_number_.npy']

import os
os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
strr=f.read().split("\n")
f.close()
print("start-------")
for i in range(len(strr)-1):
  if i%5000==0:
    print(i)
  strr[i]=strr[i].split('"values"')[1:]
  for j in range(len(strr[i])-1):
    strr[i][j]=strr[i][j][3:].split("]}]}, {")[0].split(",")
  strr[i][-1]=strr[i][-1][3:].split("]}]}]}")[0].split(",")
  strr[i]= np.array(strr[i]).astype(np.float)
  strr[i]=np.array(strr[i][0]) #cls
value=np.array(strr[:-1])
print(value.shape)
print("start clustering---------")
from thundersvm import OneClassSVM
clf = OneClassSVM(nu=nu_, kernel="rbf",gamma=gama_)#0.08 original, 0.1 for S4 case test
clf.fit(value)
predict_result = clf.predict(value)
m = 0
import collections
def second_class(strr_original, labels,threshold,flag_id):
  benign=[]
  malicious=[]
  f11=open(strr_original,'r')
  whole_words={}
  strr_original=f11.read().split("\n")
  for i in range(len(labels)):
    if labels[i]==1:
      benign.append(i)
    if labels[i]==-1:
      malicious.append(i)
  frequent_list={}
  for i in range(len(strr_original)):
    ind=i
    i=strr_original[i].split(",")[4:]
    for j in range(len(i)):
      if j not in frequent_list.keys():
        frequent_list[j]={}
      if i[j]=='':
        continue
      else:
        if i[j] not in frequent_list[j].keys():
            frequent_list[j][i[j]]=0
        if i[j] not in whole_words.keys():
            whole_words[i[j]]=0
        frequent_list[j][i[j]]=frequent_list[j][i[j]]+1
        whole_words[i[j]]=whole_words[i[j]]+1
  for i in range(len(frequent_list.keys())):
    frequent_list[i]=sorted(frequent_list[i].items(), key=lambda x: x[1], reverse=True)
    frequent_list[i]=frequent_list[i][:int(threshold*len(frequent_list[i]))]
    tmp_keys=[]
    for j in frequent_list[i]:
      tmp_keys.append(j[0])
    frequent_list[i]=tmp_keys
  for i in malicious:
    strr_tmp=strr_original[i].split(",")[4:]
    flag=False
    for j in range(len(strr_tmp)):
      if strr_tmp[j]=='':
        continue
      if strr_tmp[j] not in frequent_list[j] and whole_words[strr_tmp[j]]>8:
        flag=True
    if flag is False:
      labels[i]=1
  return labels
fpresults=[]
benign_benign=[]
for num_1 in range(len(predict_result)):
    if predict_result[num_1] == 1:
        m += 1
        benign_benign.append(str(num_1))
    else:
        fpresults.append(str(num_1))
acc = m / len(predict_result)
print("benign accuracy")
print(acc)
f=open("./embedding_data/"+suffix+"S"+str(flag)+"_test.json",'r')
strr=f.read().split("\n")
f.close()
for i in range(len(strr)-1):
  strr[i]=strr[i].split('"values"')[1:]
  for j in range(len(strr[i])-1):
    strr[i][j]=strr[i][j][3:].split("]}]}, {")[0].split(",")
  strr[i][-1]=strr[i][-1][3:].split("]}]}]}")[0].split(",")
  strr[i]= np.array(strr[i]).astype(np.float)
  strr[i]=np.array(strr[i][0]) #cls
value2=np.array(strr[:-1])

test1_time=time.time()
import numpy as np
strr=np.load('./ground_truth/'+number_list[flag-1])
labels=np.ones(len(value2))
for i in range(len(strr)):
  if int(strr[i])<=len(value2):
    labels[int(strr[i])]=-1
predict_labels = clf.predict(value2)
a1=0
a2=0
a3=0
a4=0
benign_benign=[]
fpresults=[]
benign_malicious=[]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test1')
#np.save("S"+str(flag)+"_number_benign_test.npy",benign_benign)
current_time=time.time()-test1_time
#print(current_time)
print(a1)
print(a2)
print(a3)
print(a4)
print(a1/(a1+a2))
print(a4/(a4+a3))
print(a3/(a4+a3))
print(a2/(a2+a1))
a_labels=predict_labels.copy()
for j in range(1,11):
  threshold=0.3
  predict_labels=second_class('./training_data/'+path[flag-1],a_labels,threshold,flag)
  a1=0
  a2=0
  a3=0
  a4=0
  result_array=[]
  fpresults=[]
  benign_benign=[]
  logs_f=open('./training_data/'+path[flag-1],'r')
  logs=logs_f.read().split("\n")
  logs_f.close()
  logs_classify=[]
  for i in range(len(predict_labels)):
      if labels[i]==-1 and predict_labels[i]==-1:
          a1=a1+1
      if labels[i]==-1 and predict_labels[i]==1:
          a2=a2+1
      if labels[i]==1 and predict_labels[i]==-1:
          a3=a3+1
      if labels[i]==1 and predict_labels[i]==1:
          a4=a4+1
  print('test1')
  print(threshold)
  #print(time.time()-current_time)
  current_time=time.time()
  print(a1)
  print(a2)
  print(a3)
  print(a4)
  print("TPR")
  print(a1/(a1+a2))
  print(a4/(a4+a3))
  print("FPR")
  print(a3/(a4+a3))
  print(a2/(a2+a1))
  exit()


