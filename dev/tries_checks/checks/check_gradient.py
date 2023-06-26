import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../mlgw_v1')

from GW_generator import *
from GW_helper import * 	#routines for dealing with datasets

add_feat = True

N_data = 1000
train_frac = .75
x_range = 4
N_exp = 5

#f = lambda x: x[:,0]**2+x[:,1]**3
#f = lambda x: (-0.1666*x-0.1666*x**2+x**3)
f = lambda x,y : np.arctan(x)/np.arctan(x_range) + y
f_prime_x = lambda x,y: 1./(np.arctan(x_range)*(1+np.square(x)))
f_prime_y = lambda x,y: 0.*y +1.
	#train set
#X_train_raw = np.random.uniform(-x_range,x_range,size = (N_data, 1))
X_train_raw = np.random.uniform(1.,x_range,size = (N_data, 2))
y_train = f(*X_train_raw.T)

	#test set
#X_test_raw = np.linspace(-x_range,x_range, N_data)[:,np.newaxis]
X_test_raw = np.repeat(np.linspace(1.,x_range, N_data)[:,np.newaxis],2,axis=1)
y_test = f(*X_test_raw.T)

if add_feat:
	new_feat = ["00","01"]
	X_train = add_extra_features(X_train_raw, new_feat, log_list = [0])
	X_test = add_extra_features(X_test_raw, new_feat, log_list = [0])

	##############################
#MoE starts here!!!

model_MoE = MoE_model(X_train.shape[1], N_exp)
		#opt	val_set reg verbose threshold N_it	step_size
print(X_train.shape, y_train.shape)
args = ["adam", None,   0., False,  1e-4,     None, 2e-3]
#history = model_MoE.fit(X_train, y_train, threshold = 1e-2, args = args, verbose = True)#, val_set = (test_theta, y_test))
#model_MoE.save("exp", "gat")
model_MoE.load("exp", "gat")

print(X_test.shape, X_test.ndim)
y_pred = model_MoE.predict(X_test)
y_exp = model_MoE.experts_predictions(X_test)
y_gat = model_MoE.get_gating_probs(X_test)
#print(y_gat)
print("square loss: ",np.sum(np.square(y_pred-y_test))/(y_pred.shape[0]))

#gradients
grads = np.zeros(X_test.shape)
#for i in range(X_test.shape[0]):
#	grads[i,:] = model_MoE.get_gradient(X_test[i,:])
gradients = model_MoE.get_gradient(X_test) #(N,2)

	#doing partial derivative
if add_feat == True:
	grads_old = np.divide(gradients[:,0],X_test_raw[:,0]) + np.divide(2*np.multiply(X_test[:,0], gradients[:,1]),X_test_raw[:,0]) 
	jac_transf = jac_extra_features(X_test_raw, new_feat, log_list = [0]) #(N,D+L,D)
	grads = np.zeros(grads_old.shape)
	#print(X_test_raw[20,:], X_test[20,:], jac_transf[20,:,:], gradients[20,:],np.matmul(jac_transf[20,:,:].T, gradients[20,:]) )
	grads = np.multiply(jac_transf, gradients[:,:,None]) #(N,2,1)
	#print(jac_transf.shape, grads[0])
	grads = np.sum(grads, axis =1)

print(X_test_raw[100,:], X_test[100,:], jac_extra_features(X_test_raw, new_feat, log_list = [0])[100,:,:])

plt.figure(figsize=(25, 20))

plt.plot(X_test[:,1], y_test, ls ='solid', lw = 3, c = 'k', label = 'true')
plt.plot(X_test[:,1], y_pred, ls = 'dashed', lw = 3, c='k',label = 'pred')	
plt.plot(X_test[:,1], y_gat, ls='dotted', lw=2, c = 'k',label = 'gat_pred')

plt.figure(figsize=(25, 20))
plt.plot(X_test_raw[:,0], grads[:,0], 'o', label = 'gradient')
#plt.plot(X_test_raw[:,0], grads_old, 'o', label = 'old gradient')
plt.plot(X_test_raw[:,0], f_prime_x(*X_test_raw.T), 'o', label = 'true gradient')

plt.figure(figsize=(25, 20))
plt.plot(X_test_raw[:,1], grads[:,1], 'o', label = 'gradient')
#plt.plot(X_test_raw[:,0], grads_old, 'o', label = 'old gradient')
plt.plot(X_test_raw[:,1], f_prime_y(*X_test_raw.T), 'o', label = 'true gradient')

plt.legend()


#plt.show()

#checking GW_generator gradients
gen = GW_generator("../mlgw_v1/TD_model_TEOBResumS")
theta_std = np.array([[2.,0.3,-0.2],[1.5,0.2,.6]])
theta_tilde = np.array([[20.,15.,0.3,-0.2],[23.,15,0.2,.6]])
theta = np.array([[20.,15.,0.3,-0.2, 5.6, 0.923, 1.34],[23.,15,0.2,.6,  15.6, 1.923, 2.34]])

grad_Re, grad_Im = gen.get_raw_grads(theta_std)
grad_Re, grad_Im = gen._GW_generator__grads_theta(theta_tilde, np.linspace(-8.,0.03,10000)) #basic syntax to access a "private" method
grad_Re, grad_Im = gen.get_grads(theta, np.linspace(-8.,0.03,10000))
print(grad_Re.shape)

t = np.linspace(0,1,100)
f = np.exp(t)
t_new = np.linspace(0.5,1,100)

f_new = np.interp(t_new/2.,t/3., f)
f_new_bis = np.interp(t_new,2*t/3., f)
print(np.allclose(f_new, f_new_bis))



quit()








