import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,"/home/tim.grimbergen/new_MLGW/MLGW-master/mlgw")

from mlgw.NN_model import analyse_tuner_results

#D = "/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/bayesian_tuning_22/tuning_amp_22_0123"
D = "/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/bayesian_tuning_22/tuning_ph_22_2345"

analyse_tuner_results(D, save_loc=None)
