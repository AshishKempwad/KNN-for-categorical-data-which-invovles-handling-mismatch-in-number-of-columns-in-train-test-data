#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# In[2]:


class KNNClassifier:
    def _init_(self):
        self.k=0
        self.train_data=[]
        
        
        
    def train(self,path):  
        dataset=pd.read_csv(path,header=None)
        self.train_data=dataset
        x=dataset.iloc[:,:1]
        y=dataset.iloc[:,1:]
        pixel=pd.DataFrame(y).to_numpy()
        label=pd.DataFrame(x).to_numpy()
        self.k=5
        
    
    def predict(self,path):
        test_data=pd.read_csv(path,header=None) 
        z=test_data
        test_pixel=pd.DataFrame(z).to_numpy()
        dataset=self.train_data
        x=dataset.iloc[:,:1]
        y=dataset.iloc[:,1:]
        
        
        train_data=pd.DataFrame(y)
        test_data_df=pd.DataFrame(test_data)
        train_data.drop(dataset.columns[[11]], axis = 1, inplace = True) 
        test_data_df.drop(dataset.columns[[10]], axis = 1, inplace = True) 
        
        feat=[['s','k','f','x','c','b'],['f','g','y','s'],['n','b','c','g','r','p','u','e','w','y'],['t','f'],['a','l','c','y','f','m','n','p','s'],
        ['a','d','f','n'],['c','w','d'],['b','n'],['k','n','b','h','g','r','o','p','u','e','w','y'],['e','t'],
        ['f','y','k','s'],['f','y','k','s'],['n','b','c','g','o','p','e','w','y'],['n','b','c','g','o','p','e','w','y'],
        ['p','u'],['n','o','w','y'],['n','o','t'],['c','e','f','l','n','p','s','z'],['k','n','b','h','r','o','u','w','y'],
        ['a','c','n','s','v','y'],['g','l','m','p','u','w','d'] ]
        
        df=pd.DataFrame(columns=None)
        for i in range(21):
            dummies1=pd.get_dummies(train_data.iloc[:,i],prefix='',prefix_sep='')
            dummies2=dummies1.T.reindex(feat[i]).T.fillna(0)
            df=pd.concat([df,dummies2],axis=1,sort=False)
        
        df2=pd.DataFrame(columns=None)
        for i in range(21):
            dummies3=pd.get_dummies(test_data.iloc[:,i],prefix='',prefix_sep='')
            dummies4=dummies3.T.reindex(feat[i]).T.fillna(0)
            df2=pd.concat([df2,dummies4],axis=1,sort=False)
        
        pixel=pd.DataFrame(df).to_numpy()
        label=pd.DataFrame(x).to_numpy()
        test_pixel=pd.DataFrame(df2).to_numpy()
        
        list2=[]
        for i in range(len(test_pixel)):
            list1=[]
            neighbors = []
            m=test_pixel[i]
            for j in range(len(pixel)):
                l=pixel[j]
                q=label[j]
                distance = np.linalg.norm(l-m)
                list1.append((q,distance))

            list1.sort(key=lambda ele:ele[1])
            kv=self.k
            for p in range(kv):
                neighbors.append(list1[p][0])

            output_values = [row[-1] for row in neighbors]
            prediction = max(set(output_values), key=output_values.count) 
            list2.append(prediction)
        return list2
        
        
       

