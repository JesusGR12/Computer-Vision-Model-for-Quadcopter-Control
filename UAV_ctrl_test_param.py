import numpy as np
import pandas as pd
from tensorflow import keras
import cv2

"""Data preprocessing"""
sign_mnist=pd.read_csv('sign_mnist_train.csv') #Import training dataset
sign_mnist_test=pd.read_csv('sign_mnist_test.csv') #Import testing dataset
data1=np.array(sign_mnist) #Convert to numpy arrays
data2=np.array(sign_mnist_test)
training_labels=np.reshape(data1[:,:1],(27455)) #Create training labels array
#training_labels=[x+1 for x in training_labels]
#training_labels=np.array(training_labels)
#print(training_labels)
training_images=np.reshape(data1[:,1:],(27455,28,28,1)) #Create training images array
test_labels=np.reshape(data2[:,:1],(7172)) #Create test labels array
test_images=np.reshape(data2[:,1:],(7172,28,28,1)) #Create test images array
training_images=training_images / 255.0 #Normalization
test_images=test_images/255.0 #Normalization
#exit()

i = 27455 # number of elements in training dataset
j = 7172 # number of elements in testing dataset

for iter in range(0,i):
	if training_labels[iter] == 5:
		training_labels[iter] = 1
	elif training_labels[iter] == 24:
		training_labels[iter] = 2
	elif training_labels[iter] == 2:
		training_labels[iter] = 3
	elif training_labels[iter] == 6:
		training_labels[iter] = 4
	elif training_labels[iter] == 21:
		training_labels[iter] = 5
	elif training_labels[iter] == 22:
		training_labels[iter] = 6
	else:
		training_labels[iter] = 0
		
for iter in range(0,j):
	if test_labels[iter] == 5:
		test_labels[iter] = 1
	elif test_labels[iter] == 24:
		test_labels[iter] = 2
	elif test_labels[iter] == 2:
		test_labels[iter] = 3
	elif test_labels[iter] == 6:
		test_labels[iter] = 4
	elif test_labels[iter] == 21:
		test_labels[iter] = 5
	elif test_labels[iter] == 22:
		test_labels[iter] = 6
	else:
		test_labels[iter] = 0
		
model = keras.models.load_model(r'C:\Users\jgros\Documents\CS579\Project') #Load model

p2=model.predict(test_images,batch_size=512)
print(p2.shape)
print(p2)
TP=0
TN=0
FP=0
FN=0

for iter in range(0,j):
	if (abs(1-p2[iter][1])<0.5): #margin to allow for outputs between 0.5 - 1.0
		if ((test_labels[iter]) == 1):
			TP=TP+1
		else:
			FP=FP+1
	elif (abs(1-p2[iter][2])<0.5): #margin to allow for outputs between 0.5 - 1.0
		if ((test_labels[iter]) == 2):
			TP=TP+1
		else:
			FP=FP+1
	elif (abs(1-p2[iter][3])<0.5): #margin to allow for outputs between 0.5 - 1.0
		if ((test_labels[iter]) == 3):
			TP=TP+1
		else:
			FP=FP+1
	elif (abs(1-p2[iter][4])<0.5): #margin to allow for outputs between 0.5 - 1.0
		if ((test_labels[iter]) == 4):
			TP=TP+1
		else:
			FP=FP+1
	elif (abs(1-p2[iter][5])<0.5): #margin to allow for outputs between 0.5 - 1.0
		if ((test_labels[iter]) == 5):
			TP=TP+1
		else:
			FP=FP+1
	elif (abs(1-p2[iter][6])<0.5): #margin to allow for outputs between 0.5 - 1.0
		if ((test_labels[iter]) == 6):
			TP=TP+1
		else:
			FP=FP+1
	else:
		if ((test_labels[iter]) == 0):
			TN=TN+1
		else:
			FN=FN+1

print('TP= ',TP,', TN= ',TN,', FP= ',FP,', FN= ',FN)

accuracy=(TP+TN)/j
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1=2*precision*recall/(precision+recall)

print('accuracy= ',accuracy,', precision= ',precision,', recall= ',recall,', F1= ',F1)