import lal
import matplotlib.pyplot as plt
import numpy as np
import mlgw
import scipy.optimize
import scipy.signal

from mlgw.precession_helper import angle_manager, to_J0_frame, angle_params_keeper, get_IMRPhenomTPHM_angles, get_S_effective_norm
from mlgw.GW_helper import load_dataset

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler

import joblib

###################################################################################################################################################

theta, targets, _ = load_dataset('datasets/angle_dataset_only_qs1s2t1t2phi1.dat', N_entries = 1, N_grid = None, N_data = 100_000, shuffle = False, n_params = 8)

train_fraction = 0.5
N_train = int(train_fraction*len(theta))

train_theta, train_targets = theta[:N_train], targets[:N_train]
test_theta, test_targets = theta[N_train:], targets[N_train:]

X_train = np.concatenate([train_theta, train_targets[:,[0,1,3,4]]], axis = 1)
X_test = np.concatenate([test_theta, test_targets[:,[0,1,3,4]]], axis = 1)

if True:
	K = 4
	model = GaussianMixture(K)
	model.fit(X_train)
	joblib.dump(model, 'cluster_model.gz')
else:
	model = joblib.load('NN_models/cluster_model_full_K4.gz')
	K, = model.weights_.shape

labels = model.predict(X_test)

chi_P_2D_norm, ids_s1_p, ids_s2_p = get_S_effective_norm(test_theta[:,:-1])


#Psi =  angle_params_keeper(test_targets)
scaler = RobustScaler(quantile_range=(2.0, 98.0))
#scaler = QuantileTransformer()
#scaler = MinMaxScaler()
scaler.fit(train_targets)
Psi =  angle_params_keeper(scaler.transform(test_targets))

#print(test_theta[ids_s1_p][:2]);quit()
#print(get_S_effective_norm(np.array([[1.52436715, 0.8, 0.4, 0.60446821, 2.3, 1.4, 0.5, np.nan]])));quit()

if False:
	ids_, = np.where(labels == 0)
	#ids_, = np.where(Psi.B_alpha>1)
	labels = ['q', 's1', 's2', 't1', 't2', 'phi1', 'phi2', 'fref']

	for id_1, id_2 in [(0,1), (1,2), (1,3), (2,4)]:#, (0,3), (1, np.nan), (0, np.nan)]:
		plt.figure()
		if np.isnan(id_2):
			plt.scatter(test_theta[ids_,id_1], chi_P_2D_norm[ids_], s =2)
			plt.ylabel(r'$|\chi_\perp|$')
			
		else:
			plt.scatter(test_theta[ids_,id_1], test_theta[ids_, id_2], s =2)
			plt.ylabel(labels[id_2])
		plt.xlabel(labels[id_1])

fig_beta, ax_beta = plt.subplots(1,1)
fig_alpha, ax_alpha = plt.subplots(1,1)
fig_params, ax_params = plt.subplots(1,1)
if True:
	for l in range(K):
		ids_, = np.where(labels == l)
		print(l, len(ids_))
		ax_beta.scatter(Psi.A_beta[ids_], Psi.A_alpha[ids_], s =2, label = str(l))
		ax_alpha.scatter(Psi.A_alpha[ids_], Psi.B_alpha[ids_], s =2, label = str(l))
		#ax_params.scatter(test_theta[ids_,0], chi_P_2D_norm[ids_], s =2, label = str(l))
		ax_params.scatter(test_theta[ids_,0], test_theta[ids_, 3], s =2, label = str(l))
		ax_beta.legend(); ax_alpha.legend(); ax_params.legend()
else:
	ids_, = np.where(labels >= 0)
	#ids_ = ids_s1_p
	#ids_, = np.where(test_theta[ids_s1_p,1]>0.1) #s1> 0.1 and ids_s1_p
	#ids_ = ids_s1_p[ids_]

	sc_beta = ax_beta.scatter(Psi.A_beta[ids_], Psi.A_alpha[ids_], s =2, c = chi_P_2D_norm[ids_])
	sc_alpha = ax_alpha.scatter(Psi.A_alpha[ids_], Psi.B_alpha[ids_], s =2, c = chi_P_2D_norm[ids_])
	plt.colorbar(sc_beta, ax = ax_beta)
	plt.colorbar(sc_alpha, ax = ax_alpha)

ax_beta.set_xlabel('A_beta')
ax_beta.set_ylabel('A_alpha')
ax_alpha.set_xlabel('A_alpha')
ax_alpha.set_ylabel('B_alpha')

plt.show()




