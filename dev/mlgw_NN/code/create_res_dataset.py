import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1,os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))),'mlgw'))

from NN_model_improved import create_residual_PCA
#%%
data_loc = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/22/" #provide data location on which model is trained
save_loc = "/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/res_datasets/attempt2" #provide location to save residual dataset to
model_loc = "/home/tim.grimbergen/new_MLGW/MLGW-master/dev/mlgw_NN/new_models/some_ph_models/model_5/22" #provide model to create residual dataset of

create_residual_PCA(data_loc, model_loc, save_loc, "ph", 2)