import sys
sys.path.insert(1, '../../mlgw_v2')
import matplotlib.pyplot as plt
import numpy as np
import GW_generator as gen
from ML_routines import *

#Creating a PCA dataset with cos(ph) and decomposing it
#The idea is to see how is the error: it doesn't work at all :(

g = gen.GW_generator()

K_max = 700

t = np.linspace(-20.,.2,1600)

f = np.random.uniform(.1,1.,1000)
f_test = np.random.uniform(.1,1.,10)

theta = np.random.uniform([10.,10.,-0.8,-0.8],[100.,100.,0.8,0.8], (1000,4))
theta_test = np.random.uniform([10.,10.,-0.8,-0.8],[100.,100.,0.8,0.8], (100,4))

amp, ph = g.get_modes(theta, t, out_type = "ampph")
amp_test, ph_test = g.get_modes(theta_test, t, out_type = "ampph")

X = np.cos(ph) #np.cos(np.einsum('i,j->ij', f,t))


X_test = np.cos(ph_test) #np.cos(np.einsum('i,j->ij', f_test,t))

PCA_beta = PCA_model()
E_X = PCA_beta.fit_model(X, K_max, scale_PC=True)
print("PCA eigenvalues for beta: ", E_X)

rec_X_test=  PCA_beta.reconstruct_data(PCA_beta.reduce_data(X_test))

#plt.figure()
#plt.plot(t,X.T)

plt.figure()
plt.plot(t,(rec_X_test-X_test).T)
plt.show()

