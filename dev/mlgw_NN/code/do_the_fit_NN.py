###################
#	Routine to quickly fit a single NN model, once a PCA dataset is given
###################

#A complete model relies on a collection of different modes (e.g. l,m = (2,2), (3,2) etc...).
#The fit for each mode relies on the PCA to reduce the dimensionality of the WFs and on several NNs (Neural Networks) to perform a regression from the orbital parameters to the reduced order WF.
#The ML models for each mode is stored in a dedicated folder, holding the time grid at which each WF is evaluated, the PCA models for amplitude and phase and the NN models for amplitude and phase. Each NN can be chosen to fit several PC's. Four files are outputted: a figure of training and validation loss as a function of epochs, the weights of the NN models, the features employed for basis function expansion, and the other hyperparameters of the NN with some other general information.
#A NN model (in this case for the 22 mode) is thus stored in a folder as follows:
#----22
#	----lossfunction.png 
#	----amp(ph)_PCs.h5 (weights)
#	----feat_PCs.txt
#	----Model_fit_info.txt
#
#here PCs is a list of (advisedly consecutive) integers structured as [K_1,K_2,...,] where K_i refers to the ith PC.
#Function mlgw.fit_model.create_PCA_dataset() and mlgw.fit_model.fit_NN() are helpuf to build such model, once the proper dataset are given

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))),'mlgw'))

from NN_model_improved import fit_NN, Schedulers, Optimizers, LossFunctions

lm = "22"

#I am just assuming the user already has a PCA dataset and that he does not want to create one with this script

PCA_dataset_folder = "/home/tim.grimbergen/PCA_Data/SEOBNRv4PHM_100WF_HM_16_16/{}".format(lm)		#folder in which the PCA_model is stored
#PCA_dataset_folder = '/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/res_datasets/attempt2/'
model_folder = "../new_models/SEOBNRv4PHM_models/ph_2/{}".format(lm)		#folder in which the model for the current mode must be stored

print("Dealing with {} mode".format(lm))



#specify the hyperparameters for the NN
#if a weighted loss function is used, make sure the number of weights equals the number of specified PCs
param_dict = {'layer_list' : [50,50], #a list with the number of nodes per hidden layer
              'optimizers' : Optimizers("Nadam",0.0001), #the optimizer with (initial) learning rate
              'activation' : "sigmoid", #activation function between hidden layers (default: sigmoid)
              'batch_size' : 128, #batch size
              'schedulers' : Schedulers('exponential',exp=-0.0003, min_lr = 1e-6) #how the learning rate decays during training
			 }

fit_type = "ph"
features_ = [(['q' ,'s1' ,'s2'], 2)]
epochs_ = 10000


#Here we are fitting the NN for the specified quantity (amp or phase) with the specified hyperparameters and features 
print("Saving NN model to: ", model_folder)
print("Fitting " + fit_type)
fit_NN(fit_type,
	   PCA_dataset_folder,
	   model_folder,
	   hyperparameters = param_dict,
	   N_train = None,
	   comp_to_fit = 2,
	   features = features_,
	   epochs = epochs_,
	   verbose = True,
	   residual=False #IMPORTANT: if you make a residual model, make sure this is set to True. 
)



