from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score
data=[]
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nu', action='store', default=None, dest='nu')
parser.add_argument('-gama', action='store', default=None, dest='gama')
parser.add_argument('-gpu', action='store', default=None, dest='gpu')
parser.add_argument('-thres', action='store', default=0.3, dest='thres')
args = parser.parse_args()
nu_=float(args.nu)
gama_=float(args.gama)

start_load_train=time.time()

f=open("./embedding_data/M12_benign.json",'r')
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
end_load_train=time.time()
print("end load train data")
#print(end_load_train-start_load_train)


clf = OneClassSVM(nu=nu_, kernel="rbf",gamma=gama_)
clf.fit(value)
end_train_time=time.time()
#print("training time is")
#print(end_train_time-end_load_train)



predict_result = clf.predict(value)
m = 0
print('end cluseter')
import collections
def second_class(strr_original, labels,threshold=0.3):
  benign=[]
  whole_words={}
  malicious=[]
  FNs=[]
  f11=open(strr_original,'r')
  strr_original=f11.read().split("\n")
  for i in range(len(labels)):
    if labels[i]==1:
      benign.append(i)
    if labels[i]==-1:
      malicious.append(i)
  frequent_list={}
  for i in strr_original:
    i=i.split(",")[4:]
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
        whole_words[i[j]]=whole_words[i[j]]+1
        frequent_list[j][i[j]]=frequent_list[j][i[j]]+1
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
      FNs.append(i)
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

start_load_test_time=time.time()
f=open("./embedding_data/M121_test.json",'r')
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
end_load_test_time=time.time()
print("end load test")
#print(end_load_test_time-start_load_test_time)
strr=np.load('ground_truth/M1h1_number.npy')
strr=list(strr)
labels=np.ones(len(value2))
for i in range(len(strr)):
  if int(strr[i])<=len(value2):
    labels[int(strr[i])]=-1

start_test_time=time.time()
predict_labels = clf.predict(value2)
end_test_time=time.time()
#print("end test time")
#print(end_test_time-start_test_time)
#predict_labels = np.ones(len(value2))
#for i in range(len(predict_labels)):
#  predict_labels[i]=-1

a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
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

print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)


start_cali_time=time.time()
predict_labels=second_class('./training_data/testing_preprocessed_logs_M1-CVE-2015-5122_windows_h1',predict_labels,float(args.thres))
end_cali_time=time.time()
#print("end cali  time")
#print(end_cali_time-start_cali_time)
a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
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
print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)

start_load_test_time=time.time()
f=open("./embedding_data/M122_test.json",'r')
strr=f.read().split("\n")
f.close()
for i in range(len(strr)-1):
  strr[i]=strr[i].split('"values"')[1:]
  for j in range(len(strr[i])-1):
    strr[i][j]=strr[i][j][3:].split("]}]}, {")[0].split(",")
  strr[i][-1]=strr[i][-1][3:].split("]}]}]}")[0].split(",")
  strr[i]= np.array(strr[i]).astype(np.float)
  strr[i]=np.array(strr[i][0]) #cls
  #strr[i]=np.mean(np.array(strr[i]),axis=0) #entire layer
value2=np.array(strr[:-1])
end_load_test_time=time.time()
print("end load test")
#print(end_load_test_time-start_load_test_time)
strr=np.load('ground_truth/M1h2_number.npy')
strr=list(strr)
labels=np.ones(len(value2))
for i in range(len(strr)):
  if int(strr[i])<=len(value2):
    labels[int(strr[i])]=-1
start_test_time=time.time()
predict_labels = clf.predict(value2)
end_test_time=time.time()
#print("end test time")
#print(end_test_time-end_load_test_time)
a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test2')

print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)

start_cali_time=time.time()
predict_labels=second_class('./training_data/testing_preprocessed_logs_M1-CVE-2015-5122_windows_h2',predict_labels)
end_cali_time=time.time()
#print("end cali  time")
#print(end_cali_time-start_cali_time)
a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test2')
print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)



start_load_test_time=time.time()
f=open("./embedding_data/M123_test.json",'r')
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

end_load_test_time=time.time()
print("end load test")
#print(end_load_test_time-start_load_test_time)
strr=np.load('ground_truth/M2h1_number.npy')
strr=list(strr)
labels=np.ones(len(value2))
for i in range(len(strr)):
  if int(strr[i])<=len(value2):
    labels[int(strr[i])]=-1

start_test_time=time.time()
predict_labels = clf.predict(value2)
end_test_time=time.time()
#print("end test time")
#print(end_test_time-start_test_time)
a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test3')

print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)


start_cali_time=time.time()
predict_labels=second_class('./training_data/testing_preprocessed_logs_M2-CVE-2015-5119_windows_py_h1',predict_labels)
end_cali_time=time.time()
#print("end cali  time")
#print(end_cali_time-start_cali_time)

a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test3')
print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)


start_load_test_time=time.time()
f=open("./embedding_data/M124_test.json",'r')
strr=f.read().split("\n")
f.close()
for i in range(len(strr)-1):
  strr[i]=strr[i].split('"values"')[1:]
  for j in range(len(strr[i])-1):
    strr[i][j]=strr[i][j][3:].split("]}]}, {")[0].split(",")
  strr[i][-1]=strr[i][-1][3:].split("]}]}]}")[0].split(",")
  strr[i]= np.array(strr[i]).astype(np.float)
  strr[i]=np.array(strr[i][0]) #cls
  #strr[i]=np.mean(np.array(strr[i]),axis=0) #entire layer
end_load_test_time=time.time()
print("end load test")
#print(end_load_test_time-start_load_test_time)
value2=np.array(strr[:-1])
strr=np.load('ground_truth/M2h2_number.npy')
strr=list(strr)
labels=np.ones(len(value2))
for i in range(len(strr)):
  if int(strr[i])<=len(value2):
    labels[int(strr[i])]=-1


start_test_time=time.time()
predict_labels = clf.predict(value2)
end_test_time=time.time()
#print("end test time")
#print(end_test_time-start_test_time)
a1=0
a2=0
a3=0
a4=0
fpresults=[]
benign_malicious=[]
indexes=[[],[],[],[]]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test4')
print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)


start_cali_time=time.time()
predict_labels=second_class('./training_data/testing_preprocessed_logs_M2-CVE-2015-5119_windows_py_h2',predict_labels)
end_cali_time=time.time()
#print("end cali  time")
#print(end_cali_time-start_cali_time)
a1=0
a2=0
a3=0
a4=0
benign_malicious=[]
indexes=[[],[],[],[]]
for i in range(len(predict_labels)):
    if labels[i]==-1 and predict_labels[i]==-1:
        a1=a1+1
    if labels[i]==-1 and predict_labels[i]==1:
        a2=a2+1
    if labels[i]==1 and predict_labels[i]==-1:
        a3=a3+1
    if labels[i]==1 and predict_labels[i]==1:
        a4=a4+1
print('test4')
print("TPR")
print(a1/(a1+a2))
print(a4/(a4+a3))
print("FPR")
print(a3/(a4+a3))
print(a2/(a2+a1))
print(a1)
print(a2)
print(a3)
print(a4)



exit()


