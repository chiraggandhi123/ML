import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#KNN CODE
def distance(p1,p2):
    return np.sum((p2 - p1)**2)**.5

def knn(X,Y,test,k=5):
    m = X.shape[0]
    
    d = []
    for i  in range(m):
        dist = distance(test,X[i])
        d.append((dist,Y[i]))
    
    d = np.array(sorted(d))[:,1]
    d = d[:k]
    t =  np.unique(d,return_counts=True)
    idx = np.argmax(t[1])
    pred = int(t[0][idx])
        
    return pred
################################
#Fetch face_data from FaceData/U_name.npy
#print(os.listdir('./FaceData'))
dataset_path='./FaceData'
names={}
face_data=[]
labels=[]
identity=0
#Data Loaded
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[identity]=fx[:-4]
        print("Loading data from",fx+'.npy')
        data_item=np.load(dataset_path+'/'+fx)
        face_data.append(data_item)
        target=identity*np.ones(data_item.shape[0],)
        labels.append(target)
        #print(target.shape)
        identity+=1
X=np.concatenate(face_data,axis=0)#vstack all data
Y=np.concatenate(labels,axis=0)
print(X.shape)
print(Y.shape)
#train_set=np.concatenate((X,Y),axis=1)

#We need to create the Test images

face_cascade=cv2.CascadeClassifier('/home/chirag/Desktop/machine-learning-june-2019/Lecture_03/haarcascade_frontalface_alt.xml')
cam=cv2.VideoCapture(0)
cnt=0
face_data=[]
#u_name=input()
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(gray,'chirag',(10,50),cv2.FONT_HERSHEY_COMPLEX,0,15,(255,255,255),2,cv2.LINE_AA)
        face_section=img[y-10:y+10+h,x-10:x+10+w]
        face_section=cv2.resize(face_section,(100,100))
        if cnt%10==0:
            print("Taking_frame",cnt/10)
            face_data.append(face_section)
        cnt+=1
    
        cv2.imshow('face',face_section)
    cv2.imshow('gray',img)
    k=cv2.waitKey(1)&0xFF
    if k==ord('q'):
        break
#print("TotalFace",len(face_data))
    test=np.array(face_data)
    test=test.reshape(test.shape[0],30000)
    result=knn(X,Y,test)
    print(names[result])
print(os.listdir(dataset_path))
print(names[result])
#print(face_data.shape)
#np.save('FaceData/'+u_name+'.npy',face_data)
#print('save at FaceData/'+u_name+'.npy')
#np.save()
cam.release()
#print(face_data)
cv2.destroyAllWindows()     

