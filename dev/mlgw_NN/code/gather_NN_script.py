import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))),'mlgw'))

from NN_model_improved import gather_NN

data_location = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/22/"
out_folder = "../new_models/test_full_models"
amp_model_locations = [
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_amp_models/model_1/22'
]
ph_model_locations = [
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_models/model_5/22',
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_res_models/model_2/22',
	'/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_models/model_3456/22'
]

#Needs some more testing. And need to implement the load function of mode_generator in such a way that it accepts the output file of this function as input.
gather_NN("22", data_location, amp_model_locations, ph_model_locations ,out_folder)