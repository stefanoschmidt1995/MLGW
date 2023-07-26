import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))),'mlgw'))

from mlgw.NN_model import gather_NN


mode = "55"
pca_data_location = "../pca_datasets/IMRPhenomTPHM/{}".format(mode)
out_folder = "/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/mlgw/TD_models/model_3"
amp_model_locations = [
	'../models_NN/model_IMRPhenomTPHM/amp/{}'.format(mode)
]
ph_model_locations = [
	'../models_NN/model_IMRPhenomTPHM/ph/{}/comp01'.format(mode),
	'../models_NN/model_IMRPhenomTPHM/ph/{}/comp2345'.format(mode),
]

#Needs some more testing. And need to implement the load function of mode_generator in such a way that it accepts the output file of this function as input.
gather_NN(mode, pca_data_location, amp_model_locations, ph_model_locations ,out_folder)
