import numpy as np
import sys
sys.path.insert(1, '../mlgw_v1')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
N_waves= 3000
x = np.linspace(1,100,1000)

	#creating a set of signals
f = lambda alpha: np.power(x, alpha)+ np.random.normal(0,10)

X_dataset = np.zeros((N_waves, len(x)))
alpha = np.zeros((N_waves,))
for i in range(N_waves):
	alpha[i] = np.random.uniform(-5,5)
	X_dataset[i,:] = f(alpha[i])

	#doing PCA
K_ph= 10
PCA = PCA_model()
E = PCA.fit_model(X_dataset, K_ph, scale_PC=True)
print("PCA eigenvalues: ", E)
X_red = PCA.reduce_data(X_dataset)

for k in range(K_ph):
	plt.figure(k, figsize=(15,10))
	plt.title("Component # "+str(k)+"| full dataset")#s = "+str(train_theta[0,1:]))
	plt.plot(alpha, X_red[:,k], 'o', ms = 1)
	plt.xlabel("alpha")
	plt.ylabel("PC value")

plt.figure(600)
plt.title("Phase principal components")
for i in range(7):#PCA.get_V_matrix().shape[1]):
	plt.plot(x, PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()


plt.show()


