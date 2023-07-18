import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))),'mlgw'))

from NN_model_improved import tune_model

data_loc = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/22/"

hyperparameters = { #tuple --> yields a choice in build_model
		"units" : (1,2,3,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,150,200), #units per hidden layer
		"layers" : (1,2,3,4,5,6,7,8,9,10), #num of hidden layers
		"activation" : ("sigmoid"),
		"learning_rate" : (0.000001, 0.000003, 0.00001, 0.00003, 0.0001,0.0003,0.001,0.003,0.01,0.03,0.1),
		"feature_order" : (0,1,2,3,4,5)
	}

out_folder = "/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/bayesian_tuning"

tune_model(out_folder, "tuning_[1,2,3,4]_amp" , "amp", data_loc, 4, hyperparameters,
			max_epochs = 10000, trials=100, init_trials=25)