import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,"/home/tim.grimbergen/new_MLGW/MLGW-master/mlgw")

from NN_model_improved import analyse_tuner_results

D = "/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/bayesian_tuning/tuning_[1,2,3,4]_amp/"

analyse_tuner_results(D, save_loc=D)