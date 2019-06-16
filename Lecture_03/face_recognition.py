import cv2
import numpy as np
import os

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path ="./FaceData/"
labels = []
class_id = 0
names = {}
face_data = []
labels = []

for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id]=fx[:-4]
        print("Data of file is loading",fx)
        dataitem=np.load(dataset_path+fx)
        face_data.append(dataitem)

#print(type(face_data))
#print(face_data.shape)        
face_data=np.array(face_data)
print(type(face_data))
print(face_data.shape)
print(dataitem[10].size)