import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import MLP_TF as ml
#xor inputs
x_train=np.array([[0,0],[0,1],[1,0],[1,1]])
#xor outputs
Y_train=np.array([[0],[1],[1],[0]])
n_h=2
#no.of inputs
n_x=2
#number of neurons in outpput layer
n_y=1
#number an instance of the MLP_TF class
model=ml.MLP_TF(n_x,n_h,n_y)
#Print weights before trraining
print("weights before training ",model.get_weights())

#Test trained model with  xor inputs
newdata=np.array([[1,1]])
prediction=model.predict(newdata)
print(f"Input:{newdata},predict output:{prediction}")

