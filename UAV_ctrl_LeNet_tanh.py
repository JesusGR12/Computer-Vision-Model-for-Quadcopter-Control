import numpy as np
import pandas as pd
import tensorflow as tf
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



"""From the given dataset, set output labeles."""
"""
Notation:
Command ---> Character ---> # in dataset ---> # in output

Forward ---> F ---> 5 ---> 1
Backward ---> Y ---> 24 ---> 2
Left ---> C ---> 2 ---> 3
Right ---> G ---> 6 ---> 4
Up ---> V ---> 21 ---> 5
Down ---> W ---> 22 ---> 6
Stop ---> NC ---> 0

"""
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


"""Prepare model"""
#    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
#    tf.keras.layers.MaxPooling2D(2, 2),
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, (5,5), activation='tanh', padding='same', input_shape=(28, 28, 1)),
	tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
	tf.keras.layers.Conv2D(16, (5,5), activation='tanh', padding='valid'),
	tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
	tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='tanh'),
	tf.keras.layers.Dense(84, activation='tanh'),
    tf.keras.layers.Dense(7, activation='softmax') #eight outputs, softmax layer
  ])


tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.999, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='adam')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc= ',test_acc)

model.save(r'C:\Users\jgros\Documents\CS579\Project')