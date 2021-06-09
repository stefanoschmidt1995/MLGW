###################
#	Routine to quickly fit the model, once a dataset is given
###################

#A complete model relies on a collection of different modes (e.g. l,m = (2,2), (3,2) etc...).
#The fit for each mode relies on the PCA to reduce the dimensionality of the WFs and on a MoE (Mixture of Experts) to perform a regression from the orbital parameters to the reduced order WF.
#The ML models for each mode is stored in a dedicated folder, holding the time grid at which each WF is evaluated, the PCA models for amplitude and phase and the MoE models for amplitude and phase (one for each principal component included in the analysis). The MoE models are saved as a gating function and an expert in two different files. Another file stores the features employed for basis function expansion.
#A complete model is stored in a system of folder as follow:
#	model
#	----22
#		----amp(ph)_exp#
#		----amp(ph)_gat#
#		----amp(ph)_PCA
#		----times
#	----32
#		----amp(ph)_exp#
#		----amp(ph)_gat#
#		----amp(ph)_feat
#		----amp(ph)_PCA
#		----times
#	-----....
#
#Function mlgw.fit_model.create_PCA_dataset() and mlgw.fit_model.fit_MoE() are helpuf to build such model, once the proper dataset are given

try:
	from fit_model import *
except:
	from mlgw.fit_model import *

try:
	import sys
	lm = sys.argv[1]
except:
	lm = "22" 	#mode to fit

dataset_file = "TD_datasets/IMRPhenomTPHM_dataset.{}".format(lm)	#input file for WF dataset of the mode
PCA_dataset_folder = "TD_datasets/{}_IMRPhenomTPHM".format(lm)		#folder in which to store the reduced order dataset after the PCA model is fitted
model_folder = "TD_models/model_3/{}".format(lm)		#folder in which the model for the current mode must be stored

	#control what to do
fit_PCA = True
fit_MoE_model = True

	#features to use for the basis function expansion
fifth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222", #2nd/3rd order
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222", #4th order
"00000", "00001", "00002", "00011", "00012", "00022", "00111", "00112","00122", "00222", #5th order
"01111", "01112", "01122", "01222", "02222", "11111", "11112", "11122","11222", "12222", "22222"] #5th order

fourth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222",
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]

sixth_order = fifth_order + ["000000", "000001", "000002", "000011", "000012", "000022", "000111", "000112","000122", "000222", #6th order
"001111", "001112", "001122", "001222", "002222", "011111", "011112", "011122","011222", "012222", "022222",
"111111", "111112", "111122","111222", "112222", "122222","222222"]#6th order

#seventh_order = sixth_order + [...]

no_spins = ["00", "000", "0000", "00000", "000000"]

print("Dealing with {} mode".format(lm))

if fit_PCA:
	#Here a PCA model is fitted and saved to PCA_dataset_folder. At the same time, a reduced version of the WF dataset is saved to PCA_dataset_folder
	print("Loading dataset from: ", dataset_file)
	print("Saving PCA dataset to: ", PCA_dataset_folder)
	create_PCA_dataset((4,4), dataset_file, PCA_dataset_folder, train_frac = 0.8, clean_dataset = False)

if fit_MoE_model:
	#Here many MoE models are fitted from the reduced dataset, built on PCA.
	#A MoE model is fitted for every principal component of both amplitude and phase and stored in the proper folder of the model.
	#The routines also copies the PCA models and the times to the relevant folder, making it ready to use.
	print("Saving MoE model to: ", model_folder)
	print("Fitting phase")
	fit_MoE("ph", PCA_dataset_folder, model_folder, experts = 2, comp_to_fit = None, features = sixth_order, EM_threshold = 1e-2, args = None, N_train = 14000, verbose = False, test_mismatch = True)
	print("Fitting amplitude")
	fit_MoE("amp", PCA_dataset_folder, model_folder, experts = 2, comp_to_fit = None, features = sixth_order, EM_threshold = 1e-2, args = 	None, N_train = 14000, verbose = False, test_mismatch = True)

