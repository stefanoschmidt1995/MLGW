###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from EM_MoE import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import keras

img_rows =26
img_cols = 26
num_classes = 10 #10 digits

X_train = np.load("../datasets/MNIST_data/X_train.npy")[:,1:27,1:27]
Y_train = keras.utils.to_categorical(np.load("../datasets/MNIST_data/Y_train.npy"),num_classes)
X_test = np.load("../datasets/MNIST_data/X_test.npy")[:,1:27,1:27]
Y_test = keras.utils.to_categorical(np.load("../datasets/MNIST_data/Y_test.npy"),num_classes)

X_train = np.reshape(X_train,(X_train.shape[0],img_rows*img_cols))
X_test = np.reshape(X_test, (X_test.shape[0],img_rows*img_cols))

print(X_train.shape)

softmax = softmax_regression(img_rows*img_cols, num_classes)
softmax.fit(X_train[:,:], Y_train[:,:], threshold = 1e-3, N_iter = None ,verbose = True, val_set = (X_test, Y_test))
#softmax.fit_single_loop(X_train[0:10000,:], Y_train[0:10000,:])

Y_pred= softmax.predict(X_test)

#gda = GDA(img_rows*img_cols, num_classes)
#gda.fit(X_train[0:10000,:], Y_train[0:10000,:])

#Y_pred= gda.predict(X_test, LL = True)

print(softmax.accuracy(X_test, Y_test))

#ww, pi = gda.get_weights()

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)

plt.figure(figsize=(20, 5)) 
for i in range(20):    
	ax = plt.subplot(2, 10, i + 1)    
	#plt.imshow(np.reshape(ww[i][0], (img_rows, img_cols)))
	plt.imshow(X_test[i, :, :], cmap='gray')
	#print(Y_pred[i,:])
	plt.title("Digit: {}\nPred:    {}".format(np.argmax(Y_test[i,:]), np.argmax(Y_pred[i,:])), fontsize = 18)    
	plt.axis('off')
	#matplotlib.use('pgf')
	plt.savefig('/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/tesi_latex/img/softmax_MNIST.eps', format='eps')

plt.show()






