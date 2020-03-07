"""
Module fit_model.py
===================
	Various routines for mlgw model fitting.
	Routines:
		create_PCA_dataset		routine for generating a dataset for PC projections of waves and orbital parameters. It start from a waveform dataset
		fit_MoE					routine for fitting a MoE model for regression from obrital parameters to PC projection. It takes as input a PCA dataset, and outputs many file where the model is saved.
"""

#################

import os
import sys
import warnings
import numpy as np
sys.path.insert(1, os.path.dirname(__file__)) 	#adding to path folder where mlgw package is installed (ugly?)
from shutil import copyfile #for copying files 
from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model

################# routine create_PCA_dataset
def create_PCA_dataset(K, dataset_file, out_folder, train_frac = 0.75):
	"""
create_PCA_dataset
==================
	Creates a PCA dataset starting from a waveform dataset.
	Waveform dataset can be generated with:
		mlgw.GW_helper.create_dataset_TD(N_data = 3000, N_grid = 4000, filename = "../GW_TD_dataset.dat",
		 t_coal = .4, q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
		 t_step = 1e-6, lal_approximant = "SEOBNRv2_opt")
	A PCA dataset consists in
		PCA model				a fitted PCA model in the form of mlgw.ML_routines.PCA_model (used for reduced dataset creation)
		train(test) theta		training(test) set for the orbital parameters
		train(test) amp(ph)		training(test) set for the reduced components of amplitude(phase)
		times					times at which waves are evalueted in the high dimensional representation (not useful for MoE but required by mlgw.GW_generator
	Dataset will be saved to output folder in the following files (total 9):
		"amp(ph)_PCA_model"    "PCA_train(test)_theta.dat"    "PCA_train(test)_amp(ph).dat"    "times"
	While loading, amplitude are scaled by a factor of 1e-21 to make them O(1).
	Input:
		K (tuple)		number of PC to consider (K_amp, K_ph); if int amp and ph have the same number of PC
		dataset_file	path to file holding input waveform dataset
		out_folder		output folder which all output files will be saved to.
		train_frac		fraction of data in WF datset to be included in training set (must be strictly less than 1)
	"""
	theta_vector, amp_dataset, ph_dataset, times = load_dataset(dataset_file, shuffle=True) #loading dataset
	print("Loaded datataset with shape: "+ str(ph_dataset.shape))

	train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
	train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

	print("Orbital parameters are in range: [%f,%f]x[%f,%f]x[%f,%f]"%(np.min(train_theta[:,0]), np.max(train_theta[:,0]), np.min(train_theta[:,1]), np.max(train_theta[:,1]), np.min(train_theta[:,2]), np.max(train_theta[:,2])))

	if type(K) is int:
		K = (K,K)
	if type(K) is not tuple:
		raise RuntimeError("Wrong format for number of component K. Tuple expected but got "+str(type(K)))

		#DOING PCA

		#amplitude
	PCA_amp = PCA_model()
	E_amp = PCA_amp.fit_model(train_amp, K[0], scale_PC=True)
	print("PCA eigenvalues for amplitude: ", E_amp)
	red_train_amp = PCA_amp.reduce_data(train_amp)			#(N,K) to save in train dataset 
	red_test_amp = PCA_amp.reduce_data(test_amp)			#(N,K) to save in test dataset
	rec_test_amp = PCA_amp.reconstruct_data(red_test_amp) 	#(N,D) for computing mismatch

		#phase
	PCA_ph = PCA_model()
	E_ph = PCA_ph.fit_model(train_ph, K[1], scale_PC=True)
	print("PCA eigenvalues for phase: ", E_ph)
	red_train_ph = PCA_ph.reduce_data(train_ph)			#(N,K) to save in train dataset 
	red_test_ph = PCA_ph.reduce_data(test_ph)			#(N,K) to save in test dataset
	rec_test_ph = PCA_ph.reconstruct_data(red_test_ph) 	#(N,D) for computing mismatch

	if not out_folder.endswith('/'):
		out_folder = out_folder + "/"

		#saving to files
	PCA_amp.save_model(out_folder+"amp_PCA_model")				#saving amp PCA model
	PCA_ph.save_model(out_folder+"ph_PCA_model")				#saving ph PCA model
	np.savetxt(out_folder+"PCA_train_theta.dat", train_theta)	#saving train theta
	np.savetxt(out_folder+"PCA_test_theta.dat", test_theta)		#saving test theta
	np.savetxt(out_folder+"PCA_train_amp.dat", red_train_amp)	#saving train reduced amplitudes
	np.savetxt(out_folder+"PCA_test_amp.dat", red_test_amp)		#saving test reduced amplitudes
	np.savetxt(out_folder+"PCA_train_ph.dat", red_train_ph)		#saving train reduced phases
	np.savetxt(out_folder+"PCA_test_ph.dat", red_test_ph)		#saving test reduced phases
	np.savetxt(out_folder+"times", times)						#saving times

	F_PCA = compute_mismatch(test_amp, test_ph, rec_test_amp, rec_test_ph)
	print("Average PCA mismatch: ",np.mean(F_PCA))
	
	return

################# routine fit_MoE
def fit_MoE(fit_type, in_folder, out_folder, experts, comp_to_fit = None, features = None, EM_threshold = 1e-2, args = None, N_train = None, verbose = True, train_mismatch = False, test_mismatch = True):
	"""
fit_MoE
=======
	Fit a MoE model for each component of a PCA dataset.
	It loads a PCA dataset from in_folder and fits the regression
		theta = (q,s1,s2) ---> PC_pojection(theta)
	Outputs the fitted models to out_folder
		amp(ph)_exp_#		for amplitude (phase) of expert model for PCA component #
		amp(ph)_gat_#		for amplitude (phase) of gating function for PCA component #
		amp(ph)_feat		for list of features to use for MoE models
	Furthermore it copies PCA models and times files to the out folder. Out folder will be a valid input for mlgw.GW_generator.GW_generator.
	User can choose some fitting hyperparameters.
	Input:
		fit_type ("amp","ph")	whether to fit the model for amplitude or phase
		in_folder				path to folder with the PCA dataset. It must have the format of mlgw.fit_model.create_PCA_dataset
		out_folder				path to folder to save models to. The folder can be used by mlgw.GW_generator.GW_generator
		experts ()/[]			experts to use for each PCA component fit. If int, all models have same number of experts; if list, different experts are allowed but they must match the actual number of PC in PCA dataset.
		comp_to_fit	[]			list of PCs to fit. If None, all components will be fitted. If int, it denotes the maximum PC order to be fitted.
		features []				list of feature for basis function expansion. It must be in the format of mlgw.ML_routines.add_extra_features. If None, a default second degree polynomial in q, s1, s2 will be used.
		EM_threshold ()			threshold of minumum change in LL before breaking from the EM algorithm.
		args []					list of arguments for the softmax function fit routine mlgw.EM_MoE.softmax_regression.fit. They must be in the order [optimizator, validation set, reg. constant, verbose, threshold, # iteration, step for gradient]. If None, default values are used (recommended)
		N_train					number of training points to use in the PCA dataset. If None, every point available will be used.
		verbose					whether to display EM iteration messages
		train_mismatch			whether to return mismatch and mse on train data (if True, test_mismatch = True)
		test_mismatch			whether to return mismatch and mse on test data
	Output:
		F, mse_list											average test mismatch with PCA reconstructed waves, list of mse for each of the fitted component
		F_train, F_test, mse_train_list, mse_test_list		average train (test) mismatch (if relevant). Same format as above.
	"""
	if train_mismatch:
		test_mismatch = True

	if not fit_type in ["amp","ph"]:
		raise RuntimeError("Data type for fit_type not understood. Required (\"amp\"/\"ph\") but "+str(fit_type)+" given.")
		return

	if not os.path.isdir(out_folder): #check if out_folder exists
		raise RuntimeError("Ouput folder "+str(out_folder)+" does not exist. Please, choose a valid folder.")
		return

	if not out_folder.endswith('/'):
		out_folder = out_folder + "/"
	if not in_folder.endswith('/'):
		in_folder = in_folder + "/"

	if features is None:
		features = ["00", "11","22", "01", "02", "12"]
	if type(features) is not list:
		raise RuntimeError("Features to use for regression must be given as list. Type "+str(type(features))+" given instead")
		return
	
	if type(N_train) is not int and N_train is not None:
		raise RuntimeError("Nunmber of training point to use must be be an integer. Type "+str(type(N_train))+" given instead")
		return

	if args is None:
				#opt	val_set reg   verbose threshold	N_it step
		args = ["adam", None,   1e-5, False,  1e-4,		150, 2e-3] #default arguments for sotmax fit routine

		#loading data
	train_theta = np.loadtxt(in_folder+"PCA_train_theta.dat")[:N_train,:]		#(N,3)
	test_theta = np.loadtxt(in_folder+"PCA_test_theta.dat")						#(N',3)
	PCA_train = np.loadtxt(in_folder+"PCA_train_"+fit_type+".dat")[:N_train,:]	#(N,K)
	PCA_test = np.loadtxt(in_folder+"PCA_test_"+fit_type+".dat")				#(N',K)
	PCA = PCA_model(in_folder+fit_type+"_PCA_model")							#loading PCA model

	print("Using "+str(PCA_train.shape[0])+" train data")
	
		#adding new features for basis function expansion
	train_theta = add_extra_features(train_theta, features, log_list = [0])
	test_theta = add_extra_features(test_theta, features, log_list = [0])
	D = train_theta.shape[1] #dimensionality of input space for MoE

	MoE_models = [] #list of model, one for each component
	if train_mismatch:
		PCA_train_pred = np.zeros(PCA_train.shape) #to keep values for reconstruction
	PCA_test_pred = np.zeros(PCA_test.shape) #to keep values for reconstruction

	if comp_to_fit is None:
		comp_to_fit = [i for i in range(PCA.get_dimensions()[1])]
	if type(comp_to_fit) is int:
		comp_to_fit = [i for i in range(comp_to_fit)]
	if type(comp_to_fit) is not list:
		raise RuntimeError("Components to fit must be given as list or None type. Type "+str(type(comp_to_fit))+" given instead")
		return

	if type(experts) is int:
		experts = [experts for i in comp_to_fit]

	mse_train_list = [] #list for holding mse of every PCs
	mse_test_list = [] #list for holding mse of every PCs

		#starting fit procedure
	for k in comp_to_fit:
		print("### Fitting component ", k, " | experts = ", experts[k])
			#useless variables for sake of clariness
		y_train = PCA_train[:,k]
		y_test = PCA_test[:,k]

		MoE_models.append(MoE_model(D,experts[k]))
		MoE_models[-1].fit(train_theta, y_train, threshold = EM_threshold, args = args, verbose = verbose, val_set = (test_theta, y_test))

			#doing some test
		if train_mismatch:
			y_pred = MoE_models[-1].predict(train_theta)
			mse_train_list.append( np.sum(np.square(y_pred-y_train))/(y_pred.shape[0]) )
			PCA_train_pred[:,k] = y_pred

		y_pred = MoE_models[-1].predict(test_theta)
		mse_test_list.append( np.sum(np.square(y_pred-y_test))/(y_pred.shape[0]) )
		print("Test square loss for comp "+str(k)+": ", mse_test_list[-1] )
		print("LL for comp "+str(k)+" (train,val): ", (MoE_models[-1].log_likelihood(train_theta,y_train),MoE_models[-1].log_likelihood(test_theta,y_test)))
		PCA_test_pred[:,k] = y_pred


		#saving everything to file
		#saving feature list
	outfile = open(out_folder+fit_type+"_feat", "w+")
	outfile.write("\n".join(features))
	outfile.close()
		#copying PCA model and times, for making out_folder ready to be used in 
	copyfile(in_folder+fit_type+"_PCA_model", out_folder+fit_type+"_PCA_model")
	copyfile(in_folder+"times", out_folder+"times")
		#saving MoE models
	for i in range(len(MoE_models)):
		MoE_models[i].save(out_folder+fit_type+"_exp_"+str(i),out_folder+fit_type+"_gat_"+str(i))

		#doing test
	if test_mismatch and fit_type is "ph": #testing for phase
		PCA_test_amp = np.loadtxt(in_folder+"PCA_test_amp.dat")
		PCA_amp = PCA_model(in_folder+"amp_PCA_model")
		rec_amp=PCA_amp.reconstruct_data(PCA_test_amp)
		rec_ph=PCA.reconstruct_data(PCA_test)
		rec_ph_pred=PCA.reconstruct_data(PCA_test_pred)
		F_MoE = compute_mismatch(rec_amp, rec_ph, rec_amp, rec_ph_pred)
		
	if test_mismatch and fit_type is "amp": #testing for amplitude
		PCA_test_ph = np.loadtxt(in_folder+"PCA_test_ph.dat")
		PCA_ph = PCA_model(in_folder+"ph_PCA_model")
		rec_ph=PCA_ph.reconstruct_data(PCA_test_ph)
		rec_amp=PCA.reconstruct_data(PCA_test)
		rec_amp_pred=PCA.reconstruct_data(PCA_test_pred)
		F_MoE = compute_mismatch(rec_amp, rec_ph, rec_amp_pred, rec_ph)

	if train_mismatch and fit_type is "ph": #testing for phase
		PCA_train_amp = np.loadtxt(in_folder+"PCA_train_amp.dat")[:N_train,:]
		PCA_amp = PCA_model(in_folder+"amp_PCA_model")
		rec_amp=PCA_amp.reconstruct_data(PCA_train_amp)
		rec_ph=PCA.reconstruct_data(PCA_train)
		rec_ph_pred=PCA.reconstruct_data(PCA_train_pred)
		F_MoE_train = compute_mismatch(rec_amp, rec_ph, rec_amp, rec_ph_pred)

	if train_mismatch and fit_type is "amp": #testing for amplitude
		PCA_train_ph = np.loadtxt(in_folder+"PCA_train_ph.dat")[:N_train,:]
		PCA_ph = PCA_model(in_folder+"ph_PCA_model")
		rec_ph=PCA_ph.reconstruct_data(PCA_train_ph)
		rec_amp=PCA.reconstruct_data(PCA_train)
		rec_amp_pred=PCA.reconstruct_data(PCA_train_pred)
		F_MoE_train = compute_mismatch(rec_amp, rec_ph, rec_amp_pred, rec_ph)

	if test_mismatch:
		print("Average MoE mismatch: ",np.mean(F_MoE))

	if test_mismatch and not train_mismatch:
		return np.mean(F_MoE), mse_test_list
	if train_mismatch and train_mismatch:
		return np.mean(F_MoE_train), np.mean(F_MoE), mse_train_list, mse_test_list
	return









