import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))),'mlgw'))

from NN_model_improved import check_NN_performance

D = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/22/"
amp_model_loc = [
	#'C:/Users/timgr/Documents/Scriptie/Natuurkunde/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_amp_model/22'
	]
ph_model_loc = [
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_models/model_5/22',
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_res_models/model_2/22',
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_models/model_3456/22'
	]
save_loc = ''

check_NN_performance(D, amp_model_loc, ph_model_loc, save_loc, mismatch_N=100)