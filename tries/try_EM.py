###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

from GW_helper import *
from DenseMoE import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ML_routines import *
from EM import *
from keras.models import Model
from keras.layers import Input, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#theta_vector, amp_dataset, ph_dataset, frequencies = create_dataset(10000, N_grid=512, q_max =18, spin_mag_max = 0.85, f_step=.001, f_high = 1024, f_min = None, f_max = 300, filename = "./datasets/GW_dataset.gz") #for generating dataset from scratch
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("./datasets/GW_dataset.gz", shuffle = True, N_grid = 256) #loading dataset
print("Loaded "+ str(theta_vector.shape[0])+" data")


	#splitting into train and test set
	#to make data easier to deal with
train_frac = .75
ph_scale_factor = 1. #np.std(ph_dataset) #phase must be rescaled back before computing mismatch index beacause F strongly depends on an overall phase... (why so strongly????)

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph = make_set_split(theta_vector, ph_dataset, train_frac, ph_scale_factor)

		#DOING PCA
print("#####PCA#####")
K_ph = 8 #30 apparently works well for PCA...
print("   K = ",K_ph, " | N_grid = ", test_ph.shape[1])
	#phase
ph_PCA = PCA_model()
E = ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
print("PCA eigenvalues: ", E)

red_train_ph = ph_PCA.reduce_data(train_ph)
red_test_ph = ph_PCA.reduce_data(test_ph)
rec_PCA_ph = ph_PCA.reconstruct_data(red_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_PCA_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase: ",error_ph)

F_PCA = compute_mismatch(test_amp, test_ph*ph_scale_factor, test_amp, rec_PCA_ph*ph_scale_factor)
#print("Mismatch PCA: ",F_PCA)
print("Mismatch PCA avg: ",np.mean(F_PCA))

	#TRYING EM CLUSTERING
logreg_ph = logreg_model(test_theta.shape[1],red_train_ph.shape[1], False)

red_train_ph = logreg_ph.preprocess_data(red_train_ph)[0]
red_test_ph = logreg_ph.preprocess_data(red_test_ph)[0]

	#computing covariance matrix
#print(np.cov(red_train_ph, rowvar = False)) #correlation among data can help in something??

	#setting parameters
min_comp = 0
max_comp = 8 #maximum PC to fit with GMM
print("Components to fit: ",red_train_ph[0:1,min_comp:max_comp].shape[1])
mixture_comp = 5
n_experts = 200
N_epochs = 500

	#visualizing data
plt.scatter(red_train_ph[:,0],red_train_ph[:,1])
plt.show()

N_fit_GMM = int(red_train_ph.shape[0]/20)
EM_ph = GMM(red_train_ph[0:N_fit_GMM,min_comp:max_comp], k = mixture_comp)
EM_ph.fit(tol = 5e-1) #fitting EM
EM_ph.initialize(red_train_ph[:,min_comp:max_comp])

	#fitting a model for each cluster!!
print("K range: ", min_comp,max_comp, " | n_experts: ", n_experts," | n clusters: ", mixture_comp)
for cl in range(mixture_comp): 

	train_cl_indices = EM_ph.get_cluster_indices(red_train_ph[:,min_comp:max_comp], cl)
	test_cl_indices = EM_ph.get_cluster_indices(red_test_ph[:,min_comp:max_comp], cl)

		#fitting a different MoE for each cluster
	cl_red_train_ph = red_train_ph[train_cl_indices,min_comp:max_comp]
	cl_red_test_ph = red_test_ph[test_cl_indices,min_comp:max_comp]
	cl_train_theta = train_theta[train_cl_indices,:]
	cl_test_theta = np.array(test_theta[test_cl_indices,:])
	#print(cl_test_theta.shape)

	inputs = Input(shape=(cl_train_theta.shape[1],))
	#hidden = Dense(3)(inputs)
	#hidden = DenseMoE(train_theta.shape[1], n_experts, expert_activation='linear', gating_activation='softmax')(inputs)
	hidden2 = DenseMoE(cl_red_test_ph.shape[1], n_experts, expert_activation='linear', gating_activation='softmax')(inputs)

	model = Model(inputs=inputs, outputs=hidden2)
	model.compile(optimizer = 'rmsprop', loss = 'mse')
	history = model.fit(x=cl_train_theta, y=cl_red_train_ph, batch_size=64, epochs=N_epochs, shuffle=True, verbose=0)

	#reconstructing data & mismatch for cluster 
	cl_test_ph = test_ph[test_cl_indices,:]
	cl_red_fit_ph = logreg_ph.un_preprocess_data(model.predict(cl_test_theta)) #for single model
	cl_rec_fit_ph = ph_PCA.reconstruct_data(cl_red_fit_ph)
	F = compute_mismatch(test_amp[test_cl_indices,:], cl_test_ph, test_amp[test_cl_indices,:], cl_rec_fit_ph)

	print("Cluster #",cl, " | pop: ", len(train_cl_indices)/red_train_ph.shape[0])
	print("    q (avg, std, min, max)", np.mean(cl_train_theta[:,0]),  np.std(cl_train_theta[:,0]),  np.min(cl_train_theta[:,0]),  np.max(cl_train_theta[:,0]))
	print("    a1 (avg, std, min, max)", np.mean(cl_train_theta[:,1]),  np.std(cl_train_theta[:,1]),  np.min(cl_train_theta[:,1]),  np.max(cl_train_theta[:,1]))
	print("    a2 (avg, std, min, max)", np.mean(cl_train_theta[:,2]),  np.std(cl_train_theta[:,2]),  np.min(cl_train_theta[:,2]),  np.max(cl_train_theta[:,2]))
	print("    train model loss ", model.evaluate(cl_train_theta, cl_red_train_ph, verbose=0))
	print("    test model loss ", model.evaluate(cl_test_theta, cl_red_test_ph, verbose=0))
	print("    Reconstruction error: ", np.linalg.norm(cl_test_ph - cl_rec_fit_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph)) )
	print("    Mismatch avg: ", np.mean(F))
		#plotting an example of reconstructed phase
	plt.plot(frequencies, cl_rec_fit_ph[0,:], label = 'reconstructed| '+str(cl))
	plt.plot(frequencies, cl_test_ph[0,:], label = 'true| '+str(cl))
	plt.legend()

	#computing covariance and plotting ellipses
plt.figure(0)
ax = plt.subplot(111, aspect='equal')
for cluster in range(mixture_comp):
	lambda_, v = np.linalg.eig(EM_ph.sigma_arr[cluster])
	lambda_ = np.sqrt(lambda_)
	for j in range(1, 4):
		ell = Ellipse(xy=(EM_ph.mean_arr[cluster][0,0], EM_ph.mean_arr[cluster][0,1]),
		              width=lambda_[0]*j*2, height=lambda_[1]*j*2,
		              angle=np.rad2deg(np.arccos(v[0, 0])))
		ell.set_facecolor('none')
		ell.set_facecolor('none')
		ell.set_edgecolor('b')
		ax.add_patch(ell)
	plot_cl_indices = EM_ph.get_cluster_indices(red_train_ph[:,0:max_comp], cluster)
	plt.scatter(red_train_ph[plot_cl_indices,0],red_train_ph[plot_cl_indices,1], label = 'cl '+str(cluster))
#plt.scatter(red_train_ph[:,0],red_train_ph[:,2], label= '0-2')
#plt.scatter(red_train_ph[:,0],red_train_ph[:,3], label= '0-3')
plt.legend()
plt.show()
quit()
	#reconstructing data & mismatch
cl_test_ph = test_ph[test_cl_indices,:]
cl_red_fit_ph = logreg_ph.un_preprocess_data(model.predict(cl_test_theta)) #for single model
cl_rec_fit_ph = ph_PCA.reconstruct_data(cl_red_fit_ph)

F = compute_mismatch(test_amp[test_cl_indices,:], cl_test_ph, test_amp[test_cl_indices,:], cl_rec_fit_ph)
print("Mismatch fit avg: ",np.mean(F))

quit()










x = [5,7,11,15,16,17,18]
y = [8, 5, 8, 9, 17, 18, 25]
cov = np.cov(x, y)
lambda_, v = np.linalg.eig(cov)
lambda_ = np.sqrt(lambda_)
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
ax = plt.subplot(111, aspect='equal')
for j in range(1, 4):
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), lw = 1)
    ell.set_facecolor('none')
    ell.set_edgecolor('b')
    ax.add_patch(ell)
plt.scatter(x, y)
plt.show()
quit()














