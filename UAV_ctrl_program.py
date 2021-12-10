import numpy as np
import pandas as pd
from tensorflow import keras
import cv2

model = keras.models.load_model(r'C:\Users\jgros\Documents\CS579\Project') #Load model

vid=cv2.VideoCapture(0)

while(True):

	ret,frame=vid.read()
	
	test_img=frame
	test_img=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('Test_Vid',test_img)
	test_img=cv2.resize(test_img,(28,28))
	#test_img=cv2.blur(test_img,(10,10))
	#test_img=test_img/3.0
	test_img=np.reshape(test_img,(1,28,28,1))
	test_img=test_img/255.0


	p2 = model.predict(test_img)

	max_index=np.argmax(p2)

	if max_index==1:
		print("F --> Forward")
	elif max_index==2:
		print("Y --> Backward")
	elif max_index==3:
		print("C --> Left")
	elif max_index==4:
		print("G --> Right")
	elif max_index==5:
		print("V --> Up")
	elif max_index==6:
		print("W --> Down")
	else:
		print("NC --> Not a Command --> STOP")
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
out.release()
cv2.destroyAllWindows()