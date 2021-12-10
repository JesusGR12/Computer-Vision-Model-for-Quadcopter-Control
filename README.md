# Computer-Vision-Model-for-Quadcopter-Control
Model that classifies hand gestures and issues commands to control the six-degrees-of-freedom of a quadcopter. 

The files "UAV_ctrl_LeNet_sigmoid.py", "UAV_ctrl_LeNet_relu.py" and "UAV_ctrl_LeNet_tanh.py" contain LeNet neural network models to classify images of letters from the American Sign Language alphabet. The neural networks use, respectively, sigmoid, ReLu and tanh activation functions.

The file "UAV_ctrl_model.py" contains a neural network model of custom architecture obtained from Hamdi (class lecture) for the same purpose.

The file "UAV_ctrl_test_param.py" determines the testing accuracy, precision, recall and F1 for any of the models.

The file "UAV_ctrl_program.py" takes a video output from a camera and utilizes one of the models to predict an output from a gesture made in front of the camera. Afterwards, it issues the corresponding command for the quadcopter.

To run this program: first compile one of the neural network models. Then, determine the testing parameters using "UAV_ctrl_model.py". Finally, execute "UAV_ctrl_test_param.py". The computer must have a camera. For better results, use a white background for the gestures. Only the hand performing the gestures must be visible to the camera. 

The models must be trained using the "Sign Language MNIST" dataset, availeble in: https://www.kaggle.com/datamunge/sign-language-mnist?select=sign_mnist_test. The file "amer_sign2.png", obtained from the dataset, includes the different letters of the sign language alphabet. The model takes the letters F, Y, C, G, V and W. The respective commands are "Forward", "Backward", "Left", "Right", "Up" and "Down". When no sign or an unrecognized sign is detected, the output of the model shall be "NC" or "Not a Command".
