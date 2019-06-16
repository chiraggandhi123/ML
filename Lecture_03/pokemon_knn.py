import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_1=[]
data=[]
tata=[]
tata_1=[]
result=[]
#Reading the csv files
pokemon_data=pd.read_csv('./Dataset/Pokemon/Training/train.csv')
test=pd.read_csv('./Dataset/Pokemon/Testing/test.csv')
test=np.array(test)
sample=pd.read_csv('./Dataset/Pokemon/Testing/Sample_submission.csv')
#Reading data
pokemon_data=np.array(pokemon_data)
img=os.listdir('./Dataset/Pokemon/Training/Images')
#img=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
img=np.sort(img)
img=np.array(img)
Y=pokemon_data[:,[1]]
cnt=0
#Loading Training images
for i in range(img.shape[0]):
    d=cv2.imread('./Dataset/Pokemon/Training/Images/'+img[i])
    d=cv2.resize(d,(350,350))
    data.append(d)
X=np.array(data)
#Loading Testing Images   
for i in range(test.shape[0]):
    t=cv2.imread('./Dataset/Pokemon/Testing/Images/'+test[i][0])
    t=cv2.resize(t,(350,350))
    tata.append(t)
te=np.array(tata)


#KNN CODE
def distance(p1,p2):
    return np.sum((p2 - p1)**2)**.5

def knn(X,Y,test,k=5):
    m = X.shape[0]
    result=[]
    d = []
    for i  in range(m):
        dist = distance(test,X[i])
        d.append((dist,Y[i]))
    
    d = np.array(sorted(d))[:,1]
    d = d[:k]
    t =  np.unique(d,return_counts=True)
    idx = np.argmax(t[1])
    pred = np.array(t[0][idx])
    #print(pred)
    return pred


for i in range(test.shape[0]):
    result.append(knn(X,Y,te[i],k=4))
result=np.array(result)
import csv
with open('mycsv.csv','w',newline='')as f:
    fieldnames=['ImageId','NameOfPokemon']
    thewriter=csv.DictWriter(f,fieldnames=fieldnames)
    thewriter.writeheader()
    for i in range(sample.shape[0]):
        thewriter.writerow({'ImageId':str(test[i][0]),'NameOfPokemon':str(result[i][0])})
