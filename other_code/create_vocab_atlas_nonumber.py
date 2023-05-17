import csv
import re
import time
import os
begin=time.time()
words={}
data=[]

for file in os.listdir("./"):
  file="vpnfilter.txt"
  f=open("./"+file,'r',encoding='utf-8')
  data=f.read().split('\n')
  f.close()
  strr_store=[]
  count=0
  ffflag=False
  for i in data:
      count=0
      if len(i)==0:
          strr_store.append("")
          continue
      i=re.split(' |=',i)[4:]
      tmp_data=[]
      for j in range(len(i)):
          candidate=i[j]
          #if candidate=='<' or candidate=='>':
          #  continue
          if "latency=" in candidate:
            continue 
          if len(candidate)>0 and candidate[0]=='(':
            if candidate[-1]==')' and candidate[1:-1].isdigit():
              continue
          candidate=re.split('\\\|->|_|-|\.|/|:|\(|\)',candidate)
          for kk in candidate:
              if kk==None or len(kk)==0:
                  continue
              else:
                  tmp_data.append(kk)
      strr_store.append(" ".join(tmp_data))
      for tmp_key in tmp_data:
          if tmp_key not in words.keys():
              words[tmp_key]=0
          words[tmp_key]=words[tmp_key]+1
  word_set=set()
  min_number=8
  for i in words.keys():
      if words[i]>=min_number:
          word_set.add(i)
  for i in ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]", "sim", 'sim_no', 'unlabeled','placeholder']:
    word_set.add(i)
  f1=open("./"+file+"_train_nonumber",'w',encoding='utf-8')
  f1.write("\n".join(strr_store))
  f1.close()
  f1=open("./"+file+"_vocab_nonumber",'w',encoding='utf-8')
  f1.write('\n'.join(list(word_set)))
  f1.close()
  exit()
