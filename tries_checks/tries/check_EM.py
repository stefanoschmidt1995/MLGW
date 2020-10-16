###################
#	https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
###################

import sys
sys.path.insert(1, '../mlgw_v1')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from EM_MoE import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import keras
from DenseMoE import *

N_data = 10000
D = 2
N_experts = 10
train_frac = .75

#f = lambda x: x[:,0]**2+x[:,1]**3
f = lambda x: (x[:,0]+3*x[:,1]**3+1)*5e3

x = np.random.uniform(-10,10,size = (N_data, D))
y = f(x)
y = y / np.max(np.abs(y))

#print(np.power(x[:,1],3).shape)
to_add = np.reshape(np.power(x[:,1],2), (x.shape[0],1))
#x = np.concatenate((x,to_add), axis = 1)

X_train = x[:int(train_frac*N_data),:]
X_test = x[int(train_frac*N_data):,:]
y_train = y[:int(train_frac*N_data)]
y_test = y[int(train_frac*N_data):]


	##############################
#MoE starts here!!!
	# create gating function model
gat_model = keras.Sequential()
gat_model.add(keras.layers.Dense(8, input_dim=x.shape[1], activation='sigmoid'))
#gat_model.add(keras.layers.Dense(8, activation='relu'))
gat_model.add(keras.layers.Dense(N_experts, activation='softmax'))
#gat_model.add(keras.layers.Dense(N_experts, input_dim=x.shape[1], activation='softmax'))
	# Compile model
gat_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model_MoE = MoE_model(x.shape[1], N_experts, bias = True, gating_function=gat_model)

history = model_MoE.fit(X_train, y_train, N_iter=20, threshold = None, args=[None,10,0])
#history = model_MoE.fit(X_train, y_train, N_iter=40, threshold = None, args=[40])

model_MoE.save("MoE.dat", "gat.h5")
del model_MoE
model_MoE = MoE_model(1, 1)
model_MoE.load("MoE.dat", "gat.h5", keras.models.load_model)
#model_MoE.load("MoE.dat", "gat.h5")


y_pred = model_MoE.predict(X_test)
y_exp = model_MoE.experts_predictions(X_test)
y_gat = model_MoE.get_gating_probs(X_test)
#print(y_gat)
print("square loss: ",np.sum(np.square(y_pred-y_test))/(y_pred.shape[0]))

for i in range(D):
	plt.figure(i)
	plt.plot(X_test[:,i], y_test, 'o', label = 'true')
	plt.plot(X_test[:,i], y_pred, 'o', label = 'pred')
	#plt.plot(X_test[:,i], y_exp, 'o', label = 'exp_pred')
	plt.plot(X_test[:,i], y_gat, 'o', label = 'gat_pred')
	plt.legend()

plt.show()


quit()

####################################################
	#trying with MoE layer
inputs = keras.layers.Input(shape=(x.shape[1],))
hidden2 = DenseMoE(1, N_experts, expert_activation='linear', gating_activation='softmax',
							             expert_kernel_initializer_scale=np.std(y_train))(inputs)

model = keras.Model(inputs=inputs, outputs=hidden2)
model.compile(optimizer = 'rmsprop', loss = 'mse')#tf_F_loss)
history = model.fit(x=X_train, y=y_train, batch_size=64, epochs=100, validation_split = 0.1, shuffle=True, verbose=0)
y_pred = model.predict(X_test)

####################################################
X_try = np.loadtxt("x_prova.dat")
r_try = np.loadtxt("y_prova.dat")

model = GDA(D,r_try.shape[1])
#model = softmax_regression(D,r_try.shape[1])
model.fit(X_try, r_try)
r_fit = model.predict(X_try)
print(model.accuracy(X_test,r_try))

for i in range(D):
	a = 1
	plt.figure(i)
	plt.plot(X_try[:,i], r_try[:,:], 'o', label = 'true')
	plt.plot(X_try[:,i], r_fit[:,:], 'o', label = 'pred')
	plt.legend()
plt.show()

#quit()





